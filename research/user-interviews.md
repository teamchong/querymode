# QueryMode User Interview Plan

## Goal

Validate three assumptions before building more features:
1. Zero-config onboarding is the adoption bottleneck
2. Composability (no engine boundary) is the switching reason
3. The observability features we built are what people need

Target: 5-8 interviews across two segments. 30 min each.

---

## Segments

### Segment A: "Evaluators" (3-4 people)
People who have recently evaluated or adopted a columnar query library (DuckDB, Polars, DataFusion, Arrow).

**Where to find them:**
- DuckDB Discord #showcase — people posting projects
- Polars GitHub Discussions — people asking "should I use Polars for X?"
- r/dataengineering — "DuckDB vs Polars" threads (commenters are active evaluators)
- HN "Show HN" threads for query/analytics tools (commenters who share their stack)

**Screener:**
- Have you evaluated or switched query libraries in the past 6 months?
- What's your current stack for analytical queries? (DuckDB/Polars/Pandas/Spark/raw SQL)
- Do you run queries in Node/TypeScript or Python?
- Are you building an application (not just ad-hoc analysis)?

**Disqualify if:** Pure Python/Jupyter workflow, no application context, only ad-hoc analysis.

### Segment B: "Edge/Serverless builders" (2-4 people)
People building data-intensive features on Cloudflare Workers, Vercel Edge, Deno Deploy, or similar.

**Where to find them:**
- Cloudflare Discord #workers — people asking about data access patterns
- Vercel Discussions — people hitting edge function limits with data
- X/Twitter — search "duckdb wasm" or "analytics edge function"
- Dev.to / blog posts about "serverless analytics" or "edge database"

**Screener:**
- Are you building something that queries structured data at the edge/serverless?
- What's your current approach? (API calls to a database? Embedded SQLite? DuckDB-WASM?)
- What's the data size? (< 1GB? 1-100GB? > 100GB?)
- What's the latency requirement?

**Disqualify if:** Traditional server-side only, no latency sensitivity, < 1000 rows.

---

## Interview Guide

### Opening (2 min)

> Thanks for chatting. I'm researching how developers work with query libraries — not pitching anything. There are no wrong answers. I'll ask about your experience, not about a specific tool.

### Part 1: Onboarding friction (10 min)

**Context question:**
> Walk me through the last time you evaluated a new query or data library. What did you do in the first 5 minutes?

**Follow-ups (use as needed, don't force all):**
- What was the first thing you tried to run?
- Did you use their sample data or your own data?
- Where did you get stuck, if anywhere?
- How long before you had a "this works" moment?
- Did you read docs first or jump to code?

**Probe for specifics:**
> If they mention friction: "Can you show me or describe exactly what happened? What were you looking at?"
> If they say "it was easy": "What made it easy compared to other libraries you've tried?"

**Key signal we're listening for:**
- Do they care about zero-config, or do they immediately want their own data?
- Is the friction in setup, or in understanding what the library *does*?
- Do they evaluate by running examples, reading docs, or looking at benchmarks?

### Part 2: Switching motivation (10 min)

**Context question:**
> What's the most painful thing about your current query setup? Not annoying — painful enough that you'd consider switching to something else.

**Follow-ups:**
- Have you actually switched libraries in the past year? What triggered it?
- What would you lose if you switched? What keeps you on your current tool?
- If you could change one thing about [their tool], what would it be?

**Probe for the "no engine boundary" value prop:**
> Have you ever needed to run custom logic — like a scoring function or business rule — in the middle of a query pipeline? What did you do?

**If they've used UDFs:**
> How was that experience? What was awkward about it?

**If they haven't:**
> How do you handle the gap between "query results" and "business logic that uses those results"?

**Key signal:**
- Is "composability" / "no boundary" a real pain they feel, or a theoretical concern?
- What's the actual switching trigger? Performance? API ergonomics? Deployment flexibility?
- Do they even think in terms of "operator pipelines" or is that our framing?

### Part 3: Observability needs (5 min)

**Context question:**
> Show me (or describe) how you debug a slow query today. What do you look at first?

**Follow-ups:**
- Do you have any monitoring or logging for query performance?
- When a query is slow, what's your first hypothesis? (bad filter? too much data? missing index?)
- Have you ever used EXPLAIN or query plan visualization?
- If you could see one number after every query, what would it be?

**Probe for format preference:**
> If we gave you a one-liner like `"20 rows in 3.2ms | 847 pages skipped | 12KB read"` after every query, would that be useful? What would you change about it?

**Key signal:**
- Do they debug queries at all, or just rewrite them?
- Do they want pre-execution estimates or post-execution summaries?
- Would they integrate with existing observability (OTel, Datadog) or use standalone formatters?

### Closing (3 min)

> If you were building a query library from scratch for your use case, what's the #1 thing you'd make sure it does well?

> Is there anything I didn't ask about that matters to you?

> Can I follow up if I have one more question? (get permission for async follow-up)

---

## After each interview

Fill in the tracking sheet (research/interview-tracker.md):
- Participant ID (P1, P2, ...)
- Segment (A or B)
- Current stack
- Key quotes (verbatim, not paraphrased)
- Onboarding: what they do first, where they get stuck
- Switching: what would make them switch, what holds them back
- Observability: what they look at, what they wish they had
- Surprise: anything unexpected

---

## Outreach templates

### Cold DM (Discord/Twitter)

> Hey — I'm researching how devs pick and use query libraries (DuckDB, Polars, etc). Not selling anything. Would you be up for a 30-min chat about your experience? I'll share what I learn with the community afterward.

### Forum/thread reply

> Interesting setup. I'm doing user research on query library adoption — specifically what makes people switch (or not). Would you be open to a quick 30-min chat? DM me if interested.

### Follow-up after agreement

> Thanks! Here's a Calendly/time link: [LINK]
> 
> Quick context: I'll ask about your experience evaluating/using query libraries — what worked, what didn't, what you wish existed. No prep needed. I'll share anonymized findings afterward if you're interested.

---

## Decision framework

After 5+ interviews, synthesize into:

| Assumption | Validated? | Evidence | Action |
|-----------|-----------|----------|--------|
| Zero-config is the onboarding bottleneck | Yes/No/Partial | Quotes + patterns | Keep/kill/pivot `demo()`/`fromJSON()` |
| Composability is the switching reason | Yes/No/Partial | Quotes + patterns | Keep/kill/pivot examples + messaging |
| Our observability features match needs | Yes/No/Partial | Quotes + patterns | Keep/kill/pivot formatters + timing |

**Decision rules:**
- If 4/5+ people validate → double down, ship prominently
- If 2-3/5 validate → keep but don't prioritize, dig deeper on the alternative
- If 0-1/5 validate → kill or radically rethink

What we learn should directly change the README messaging, feature priority, and what we build next.
