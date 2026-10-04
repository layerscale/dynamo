// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A thread that exits while a large backlog of expired CRTC epoch garbage is queued must
//! not run that garbage's destructors recursively.
//!
//! crossbeam-epoch collects during a thread's final unpin, and any pin inside a deferred
//! destructor at that point registers a fresh participant whose first pin collects again,
//! nesting once per expired bag. This runs in its own test binary so no other test shares
//! the process-global collector.

use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use dynamo_kv_router::ConcurrentRadixTreeCompressed;
use dynamo_kv_router::indexer::{SyncIndexer, WorkerTask};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
};

const TOP_NODES: u64 = 20_000;

/// Stores `blocks` under `parent`. Every block id is unique here, so it doubles as the
/// sequence hash of the prefix it ends.
fn store(parent: Option<u64>, blocks: &[u64]) -> WorkerTask {
    let blocks = blocks
        .iter()
        .map(|&block| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(block),
            tokens_hash: LocalBlockHash(block),
            mm_extra_info: None,
        })
        .collect();
    WorkerTask::Event(RouterEvent::new(
        0,
        KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent.map(ExternalSequenceBlockHash),
                start_position: None,
                blocks,
            }),
            dp_rank: 0,
        },
    ))
}

#[test]
fn thread_exit_with_expired_garbage_backlog_does_not_overflow() {
    let tree = Arc::new(ConcurrentRadixTreeCompressed::new());
    let (events, receiver) = flume::unbounded();
    let worker = {
        let tree = tree.clone();
        thread::spawn(move || tree.worker(receiver, None).unwrap())
    };
    for i in 0..TOP_NODES {
        let base = i * 4 + 1;
        // A store diverging after `base` splits [base, base + 1] into [base] -> {[base + 1],
        // [base + 2]}, so every top node owns a compact child map.
        events.send(store(None, &[base, base + 1])).unwrap();
        events.send(store(Some(base), &[base + 2])).unwrap();
    }
    events.send(WorkerTask::Terminate).unwrap();
    worker.join().unwrap();

    // A thread pinned during teardown keeps every snapshot the drop retires from expiring.
    let (pinned_tx, pinned_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let staller = thread::spawn(move || {
        let _guard = crossbeam_epoch::pin();
        pinned_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    pinned_rx.recv().unwrap();
    drop(tree);
    release_tx.send(()).unwrap();
    staller.join().unwrap();

    // Let the backlog expire without collecting most of it.
    for _ in 0..2 {
        crossbeam_epoch::pin().flush();
    }

    // The exiting thread's final unpin collects, because it has pinned 128 times.
    thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            for _ in 0..128 {
                drop(crossbeam_epoch::pin());
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
