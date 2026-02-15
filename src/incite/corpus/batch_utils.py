"""Shared batch API polling utilities."""

import time


def poll_batch(client, batch_id: str, poll_interval: int = 30) -> None:
    """Poll an Anthropic batch until processing completes.

    Args:
        client: Anthropic client (sync)
        batch_id: Batch ID to poll
        poll_interval: Seconds between status checks
    """
    batch = client.messages.batches.retrieve(batch_id)
    while batch.processing_status != "ended":
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = (
            counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        )
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        print(
            f"  Status: {batch.processing_status} | "
            f"Done: {done}/{total} | "
            f"Succeeded: {counts.succeeded} | "
            f"Errors: {counts.errored}"
        )
