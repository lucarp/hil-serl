# A degradation loop between operator rescue and a finite online buffer

**Observed twice on real hardware (SO-101, HIL-SERL, LeRobot).** A policy improves,
plateaus, then degrades — and the degradation is driven by the operator's own
corrections interacting with a fixed-size replay buffer. Recorded here because it is
not described in the HIL-SERL paper, it is invisible to the metrics the pipeline logs,
and the natural human response makes it worse.

## The observation

Cube-lift run, measured from the learner's own replay-buffer dumps (each episode
labelled with `complementary_info.is_intervention` per frame):

| window | intervention | autonomous episodes | autonomous successes | mean length | failures |
|---|---|---|---|---|---|
| ep 188–202 | 31% | 7 | 6 | 50 steps | 2 |
| ep 203–217 | 44% | 3 | 3 | 61 | 2 |
| ep 218–232 | 63% | 2 | 2 | 67 | 0 |
| ep 233–247 | 65% | **0** | **0** | 76 | 4 |

Intervention doubled, autonomous episodes went to zero, and episodes lengthened toward
the timeout as the policy stalled waiting for help. The same shape had already occurred
in the insertion run: success peaked at 80% around episode 75–99, then fell to 44% while
intervention rose from 50% to 60%.

## The mechanism

1. The policy falters — a bad patch, a new cube position, ordinary RL variance.
2. The operator intervenes more, because watching a robot fail is uncomfortable and
   because intervening is what the method asks for.
3. Every intervention frame is written to **both** buffers (by design — interventions
   are demonstrations in HIL-SERL).
4. The online buffer is finite (`online_buffer_capacity`, 15 000 frames ≈ 247 episodes
   here). As heavily-intervened episodes accumulate, the **autonomous** episodes —
   the only record of the policy succeeding on its own — are evicted first, being oldest.
5. RLPD samples 50% of every batch from that buffer. The policy's own successful
   experience is now absent from training, so it reverts to depending on the operator.
6. Which produces more intervention. Return to 2.

The loop is self-reinforcing and the operator is inside it. Nothing errors; throughput,
losses and gradient norms all look normal throughout.

## Why the headline metrics hide it

- **Episode reward stays high** — the operator is rescuing episodes, so success rate can
  *rise* while the policy gets worse. In the window above, overall success read 90% while
  autonomous success was 0%.
- **Block averaging masks onset.** Averaged over 41 episodes, the same run reported
  "36% intervention, 12 autonomous, 11 succeeded" — mixing the strong early part of the
  block with the collapse at its end. The decline was only visible in windows of ~15.
- **`loss_critic` says nothing.** With a sparse binary reward it sits near 1e-3 whether
  learning or not; most transitions genuinely have zero reward and near-zero TD error.

**The metric that does work:** count episodes with <5% intervention, and of those, how
many reached reward. Computed from the buffer dump, not from any logged scalar.

## A second, related eviction

`offline_buffer_capacity` defaulted to 3 000 against a 1 913-frame demo set. Since
interventions are also written to the offline buffer, ~1 087 intervention frames were
enough to wrap the ring buffer and begin overwriting the **original demonstrations** —
the fixed anchor RLPD depends on. Confirmed by the dump sitting at exactly 3 000 frames.
Raising it to 15 000 stops further eviction but does not restore what was already lost.

So both halves of every batch can quietly drift toward "recent interventions": the
offline half by overwriting the demos, the online half by evicting autonomous episodes.

## Mitigations

- **Behavioural, free:** when the policy starts failing, stop rescuing. Let failed
  episodes end as failures. The instinct to help is what feeds the loop.
- **Size the offline buffer** well above the demo set so demonstrations are never evicted.
- **Promote the policy's own autonomous successes into the offline buffer**, where nothing
  evicts them, so its best behaviour becomes a permanent anchor rather than something that
  scrolls off the end of a FIFO.
- **A larger online buffer** delays the loop but does not break it.
- **Monitor autonomous success in short windows (~15 episodes).** By the time a 40-episode
  average moves, the buffer has already turned over.

## Status

Observed, measured, twice. The mechanism above is a hypothesis consistent with all the
data collected so far; it has not been tested by a controlled experiment. The obvious
test is an A/B: identical runs from the same checkpoint, one where the operator rescues
a failing policy and one where failures are allowed to stand.
