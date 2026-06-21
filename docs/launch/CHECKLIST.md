# Launch checklist

Drafts live in this folder. Nothing posts automatically — you post.

## Pre-flight (do these first)
- [ ] PR #1 merged to `main`; CI green on `main`.
- [ ] Tag `v0.6.0` pushed.
- [ ] (Optional) `loopllm 0.6.0` published to PyPI so `pip install loopllm` works.
- [ ] Record a 15–30s IDE clip (Cursor/VS Code) of `loop_start → step → end`
      using `docs/demo/agent_loop_demo.md`; drop it in `img/` and link from the
      X thread. Existing screenshots in `img/` are good supporting stills.
- [ ] Re-run `python benchmarks/adaptive_vs_fixed.py` and confirm the README
      numbers still match.
- [ ] Skim the README top-to-bottom on GitHub (no duplicate sections, badges
      resolve, links work).

## Timing
- Best windows (US audience): Tue–Thu, ~8–10am ET.
- Post **Show HN** and the **X thread** within the same hour; cross-link once
  traction starts. Reddit (r/LocalLLaMA) a few hours later, not simultaneously.

## Show HN
- Title + body: `docs/launch/show-hn.md`.
- Submit the GitHub repo URL (not a blog post).
- Be online for the first 2–3 hours to answer comments.

## X / Twitter
- Thread: `docs/launch/twitter-thread.md`. Lead tweet must have the visual.

## Reddit
- r/LocalLLaMA + r/MachineLearning drafts: `docs/launch/reddit.md`.
- Read each subreddit's self-promo rules; lead with substance, link last.

## First-comment template (pin under Show HN)

> Author here. Quick FAQ:
>
> **Why not just let the LLM decide when it's done?** That works until it
> doesn't — it loops or quits early, and you can't budget cost ahead of time.
> This predicts a step budget up front and gives an interpretable stop reason.
>
> **Why not RL / a trained model?** No training data, decisions are
> interpretable (every stop prints `E[delta]=…±…, threshold=…`), O(1) memory per
> arm, and it degrades to sensible task-type defaults during cold start.
>
> **Is the benchmark cherry-picked?** It's a simulation with stated assumptions
> (diminishing-returns curves + noise); it measures decision efficiency given a
> quality signal, not absolute model quality. Code is in `benchmarks/`, seed
> fixed — please poke at it.
>
> **What's the catch?** The progress score is whatever signal you feed it
> (self-rating or an evaluator), so garbage-in applies. It advises *your* loop;
> it doesn't run the agent for you.
