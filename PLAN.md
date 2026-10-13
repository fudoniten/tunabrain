# Tunabrain plan

Last reviewed: 2026-10-13

This is the one place for ongoing work on Tunabrain: what's being worked on,
what's next, known issues, and the decisions that still shape the code. Delete
items once they're done (git history keeps them). Cross-repo priorities live
in the [ecosystem plan](https://github.com/fudoniten/pseudovision/blob/master/PLAN.md#ecosystem).

## Now

- [ ] Delete endpoints nobody calls (ecosystem step 3): `/channel-mapping`
      (and its chain and models), and the strategy endpoints
      `/api/scheduling/get-{quarterly,monthly}-strategy` with
      `scheduling/quarterly_strategy.py`, `monthly_strategy.py`, and the
      placeholder estimates in `cost.py`.
- [ ] Support Grout's switch to `/enrich/long-form` for `program` items
      (Grout owns the change; check the endpoint holds up on real long-form
      clips and the timeout fits).
- [ ] Add GitHub CI that runs `nix flake check` (ecosystem step 4).

## Next

- [ ] Move `scheduling/expander.py` out of `src/` (under `tests/` or a
      `reference/` package). It's only the reference implementation for the
      golden tests; TS runs the production expander.

## Later

- A TTS endpoint for voiced bumpers, if wanted. TS's TTS stub is being
  deleted, and any voice work belongs here.
- Settle on kebab-case at the HTTP boundary to match PV and Grout, if the
  ecosystem standardizes on one casing.
- Auth on the endpoints once development winds down.

## Known issues

- `/enrich/long-form` depends on the cluster's whisper-http and subgen
  services and on `ffmpeg`; it hasn't had a production caller yet.

## Decisions

- 2026-07-07: No generic chat-completions endpoint. Every caller need gets a
  typed, purpose-built endpoint.
- All AI calls in the ecosystem go through Tunabrain; callers hold no model
  keys.
- Tunabrain is stateless. Grounding context that should persist (e.g. a
  human-corrected summary) is stored by the caller and sent back.
