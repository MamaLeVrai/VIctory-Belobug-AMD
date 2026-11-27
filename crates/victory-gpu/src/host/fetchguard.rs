//! With the relay's fetch on the copy queue (`relay_copy`): the fetch never
//! runs beside drawing-queue work that writes the bytes it reads.
//!
//! A copy-queue fetch of a landed flight's pieces used to run beside the
//! submissions sent after that flight, which may write the same card copies
//! (their shaders' UAV writes, the sync list's sends of title bytes). D3D12
//! allows one queue to write a buffer while another reads it only where the
//! two do not touch the same bytes; here they did, and the test
//! `a_result_caught_in_between_is_put_right` caught the fetch in the middle
//! of the next flight 12 times in 12 without `VICTORY_FETCH_CHECK`. On RDNA 2
//! cards the drawing queue and the high-priority copy queue at the same bytes
//! is the one cross-queue access the relay adds (issue #376: device resets
//! with the relay on, none without).
//!
//! Two rules make the access never overlap, by the byte (offsets of a copy),
//! not by the whole copy: a submission that writes one part of a big copy
//! does not hold back the fetch of another part of it (#428: by whole copies,
//! on an RX 7900 XT, a fifth of the fetches went to the drawing queue).
//!
//! - the ranges a submission writes of each copy are counted from its
//!   `Host::card_before` (before anything of the submission goes to the
//!   drawing queue) until the submission's plan is dropped (`card_done`,
//!   after it ran); a fetch with a piece that shares a byte with a range
//!   another submission in the air writes goes to the drawing queue, behind
//!   that submission, as without `VICTORY_RELAY_COPY`;
//! - a fetch that goes to the copy queue names its pieces here before it is
//!   sent, and a submission counted while it is on its way that writes a
//!   byte of one of them makes the drawing queue wait for it (a wait on the
//!   card, not on the CPU) before its first list.
//!
//! Both take the one lock, so a submission is counted before a fetch looks,
//! or waits for the fetch.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::{ID3D12CommandQueue, ID3D12Fence};

/// Offsets `[from, to)` of a copy.
pub(super) type Span = (usize, usize);

/// All of a copy: for writes not known by the byte.
pub(super) const WHOLE: Span = (0, usize::MAX);

#[derive(Default)]
struct Guard {
    /// By copy ([`crate::shadow::Shadow::id`]): of each submission not done
    /// with that writes it on the card, its number and the ranges, sorted
    /// and joined.
    writers: HashMap<u64, Vec<(u64, Vec<Span>)>>,
    /// The last number given.
    numbered: u64,
    /// The pieces of the fetch on the copy queue, by copy, sorted and
    /// joined, the fence it signals and the value.
    fetching: HashMap<u64, Vec<Span>>,
    fence: Option<ID3D12Fence>,
    value: u64,
}

// SAFETY: free-threaded D3D12 objects, under the lock.
unsafe impl Send for Guard {}

static GUARD: Mutex<Option<Guard>> = Mutex::new(None);

/// Since the last frame-rate line: fetches on the copy queue, fetches sent
/// to the drawing queue instead, waits of the drawing queue for a fetch.
pub(super) static STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Of the fetches sent to the drawing queue, those held back only by a write
/// of a whole copy (a copy a submission merely looks at: [`WHOLE`]), and
/// those held back by a write of known bytes. Since the last frame line.
pub(super) static HELD: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Sorted, the empty ones out, those that overlap or touch made one.
fn joined(mut spans: Vec<Span>) -> Vec<Span> {
    spans.retain(|s| s.0 < s.1);
    spans.sort_unstable();
    let mut out: Vec<Span> = Vec::with_capacity(spans.len());
    for (from, to) in spans {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Whether `[from, to)` shares a byte with one of `spans` (sorted, joined).
fn meets(spans: &[Span], (from, to): Span) -> bool {
    let i = spans.partition_point(|s| s.1 <= from);
    spans.get(i).is_some_and(|s| s.0 < to)
}

/// Whether two sorted, joined lists share a byte.
fn meet(a: &[Span], b: &[Span]) -> bool {
    let (short, long) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    short.iter().any(|&s| meets(long, s))
}

/// The ranges one submission writes on the card, counted until it is
/// dropped.
pub(super) struct Writes {
    number: u64,
    copies: Vec<u64>,
}

impl Drop for Writes {
    fn drop(&mut self) {
        let mut guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let Some(guard) = guard.as_mut() else {
            return;
        };
        for id in &self.copies {
            if let Some(all) = guard.writers.get_mut(id) {
                all.retain(|w| w.0 != self.number);
                if all.is_empty() {
                    guard.writers.remove(id);
                }
            }
        }
    }
}

/// Whether the guard is on (`VICTORY_RELAY_FETCH_GUARD=1`; off by default).
/// Measured on an RX 7900 XT when it went by whole copies it cost ~0.7 ms of
/// fetch wait a frame and more stalls before the title's first list call,
/// for an overlap no game run has been shown to suffer from (#428): on for
/// whoever sees the wrong data. By the byte since; not measured in a game.
pub(super) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RELAY_FETCH_GUARD").is_some_and(|v| v == "1"))
}

/// A submission is about to write `ranges` of copies (a copy may come
/// several times) on the card: counted, and if a fetch on the copy queue is
/// reading a byte of them, `drawing` waits for it before anything sent to it
/// after this call.
pub(super) fn writing(drawing: &ID3D12CommandQueue, ranges: Vec<(u64, Vec<Span>)>) -> Writes {
    let mut by: HashMap<u64, Vec<Span>> = HashMap::new();
    for (id, spans) in ranges {
        by.entry(id).or_default().extend(spans);
    }
    let mut guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let guard = guard.get_or_insert_with(Guard::default);
    guard.numbered += 1;
    let number = guard.numbered;
    let mut copies = Vec::with_capacity(by.len());
    let mut wait = false;
    for (id, spans) in by {
        let spans = joined(spans);
        if spans.is_empty() {
            continue;
        }
        wait = wait || guard.fetching.get(&id).is_some_and(|f| meet(f, &spans));
        guard.writers.entry(id).or_default().push((number, spans));
        copies.push(id);
    }
    if wait && let Some(fence) = &guard.fence {
        // SAFETY: a live queue and fence; the value is signalled behind the
        // fetch, which waits for nothing.
        if unsafe { drawing.Wait(fence, guard.value) }.is_ok() {
            STATS[2].fetch_add(1, Relaxed);
        } else {
            super::fail(format_args!("the drawing queue could not wait for a fetch on the copy queue"));
        }
    }
    Writes { number, copies }
}

/// A fetch of `pieces` (copy and offsets; each may come several times) for
/// the flights whose plans' counts are `landing`: whether it may go to the
/// copy queue. If so it is named here, with the value it will signal on
/// `fence`; else a submission still in the air writes a byte of one of them,
/// and it goes to the drawing queue.
pub(super) fn may_copy<'a>(
    pieces: impl Iterator<Item = (u64, usize, usize)>,
    landing: impl Iterator<Item = &'a Writes>,
    fence: &ID3D12Fence,
    value: u64,
) -> bool {
    let ours: HashSet<u64> = landing.map(|w| w.number).collect();
    let mut by: HashMap<u64, Vec<Span>> = HashMap::new();
    for (id, from, to) in pieces {
        by.entry(id).or_default().push((from, to));
    }
    let by: HashMap<u64, Vec<Span>> = by.into_iter().map(|(id, spans)| (id, joined(spans))).collect();
    let mut guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let guard = guard.get_or_insert_with(Guard::default);
    let others = by.iter().any(|(id, spans)| {
        guard
            .writers
            .get(id)
            .is_some_and(|all| all.iter().any(|(number, written)| !ours.contains(number) && meet(spans, written)))
    });
    if others {
        STATS[1].fetch_add(1, Relaxed);
        // Whether some write of known bytes meets a piece: else the whole
        // copy of a submission that merely looks at it held the fetch back.
        let known = by.iter().any(|(id, spans)| {
            guard.writers.get(id).is_some_and(|all| {
                all.iter().any(|(number, written)| {
                    !ours.contains(number)
                        && written.iter().all(|w| *w != WHOLE)
                        && meet(spans, written)
                })
            })
        });
        HELD[usize::from(known)].fetch_add(1, Relaxed);
        return false;
    }
    guard.fetching = by;
    guard.fence = Some(fence.clone());
    guard.value = value;
    STATS[0].fetch_add(1, Relaxed);
    true
}

/// The fetch that signals `value` has run.
pub(super) fn fetched(value: u64) {
    let mut guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(guard) = guard.as_mut()
        && guard.value == value
    {
        guard.fetching.clear();
        guard.fence = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::D3D12_FENCE_FLAG_NONE;

    /// The tests here share the guard.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    /// Joined lists meet where they share a byte, not where they only touch.
    #[test]
    fn ranges_meet_only_where_they_share_a_byte() {
        let a = joined(vec![(30, 40), (0, 10), (10, 20), (5, 5)]);
        assert_eq!(a, [(0, 20), (30, 40)]);
        assert!(!meets(&a, (20, 30)), "between");
        assert!(meets(&a, (19, 21)) && meets(&a, (39, 50)) && meets(&a, (25, 31)));
        assert!(!meets(&a, (40, 50)), "touching the end");
        assert!(meet(&a, &[WHOLE]) && !meet(&a, &[(20, 30), (40, 99)]));
    }

    /// Counts come and go with the submissions; a fetch of a copy a
    /// submission in the air writes is refused, of one only landing
    /// flights write is taken.
    #[test]
    fn a_fetch_goes_to_the_copy_queue_only_when_nothing_in_the_air_writes_it() {
        let Some(host) = super::super::host() else {
            return;
        };
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: plain object creation.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        // (Ids no copy has: the counts are the test's own.)
        let (a, b) = (u64::MAX - 7, u64::MAX - 8);
        let landing = writing(&host.queue, vec![(a, vec![WHOLE])]);
        let later = writing(&host.queue, vec![(a, vec![WHOLE]), (a, vec![(0, 4)])]);
        assert!(!may_copy([(a, 0, 64)].into_iter(), [&landing].into_iter(), &fence, 1), "a later submission writes it");
        drop(later);
        assert!(
            may_copy([(a, 0, 64), (a, 0, 64)].into_iter(), [&landing].into_iter(), &fence, 2),
            "only the landing flight writes it"
        );
        // While it is on its way a submission that writes it makes the
        // drawing queue wait (the value is signalled first: other tests
        // share the queue, it must not stand).
        // SAFETY: a live fence.
        unsafe { fence.Signal(2) }.unwrap();
        let waits = STATS[2].load(Relaxed);
        let during = writing(&host.queue, vec![(a, vec![(8, 9)]), (b, vec![WHOLE])]);
        assert!(STATS[2].load(Relaxed) > waits, "the drawing queue waits for the fetch");
        fetched(2);
        assert!(GUARD.lock().unwrap().as_ref().unwrap().fence.is_none());
        drop((landing, during));
        let guard = GUARD.lock().unwrap();
        assert!(!guard.as_ref().unwrap().writers.contains_key(&a) && !guard.as_ref().unwrap().writers.contains_key(&b));
    }

    /// By the byte: a submission in the air that writes another part of the
    /// copy lets the fetch go to the copy queue, and the drawing queue does
    /// not wait for it; one that writes a byte of a piece fetched sends the
    /// fetch to the drawing queue, or makes the drawing queue wait.
    #[test]
    fn only_a_write_that_overlaps_a_piece_holds_the_fetch_back() {
        let Some(host) = super::super::host() else {
            return;
        };
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: plain object creation.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let c = u64::MAX - 9;
        const PIECE: usize = 1 << 16;
        // The landing flight wrote piece 0 and 2; a later one in the air
        // writes piece 1 of the same copy.
        let landing = writing(&host.queue, vec![(c, vec![(0, PIECE), (2 * PIECE, 3 * PIECE)])]);
        let beside = writing(&host.queue, vec![(c, vec![(PIECE, 2 * PIECE)])]);
        let pieces = || [(c, 0, PIECE), (c, 2 * PIECE, 3 * PIECE)].into_iter();
        let refused = STATS[1].load(Relaxed);
        assert!(may_copy(pieces(), [&landing].into_iter(), &fence, 11), "another part of the copy: the copy queue");
        assert_eq!(STATS[1].load(Relaxed), refused);
        // On its way: a write of the part between does not make the drawing
        // queue wait, one at a byte of piece 2 does.
        // SAFETY: a live fence.
        unsafe { fence.Signal(11) }.unwrap();
        let waits = STATS[2].load(Relaxed);
        let between = writing(&host.queue, vec![(c, vec![(PIECE + 100, 2 * PIECE)])]);
        assert_eq!(STATS[2].load(Relaxed), waits, "no wait for a write beside the fetch");
        let over = writing(&host.queue, vec![(c, vec![(3 * PIECE - 1, 3 * PIECE + 8)])]);
        assert_eq!(STATS[2].load(Relaxed), waits + 1, "a wait for a write into a piece fetched");
        fetched(11);
        // A later flight at a byte of a piece: the drawing queue.
        assert!(!may_copy(pieces(), [&landing].into_iter(), &fence, 12), "a write into a piece: the drawing queue");
        assert_eq!(STATS[1].load(Relaxed), refused + 1);
        drop(over);
        assert!(may_copy(pieces(), [&landing].into_iter(), &fence, 13), "that one done: the copy queue again");
        fetched(13);
        drop((landing, beside, between));
        assert!(!GUARD.lock().unwrap().as_ref().unwrap().writers.contains_key(&c));
    }
}
