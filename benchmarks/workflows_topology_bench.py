#!/usr/bin/env python3
"""Memory Topology Routing Benchmark.

Measures the retrieval precision and recall improvement from namespace/room
scoping compared to unscoped (global) hybrid search.

Scenario
--------
A multi-team knowledge base with 90 facts distributed across 6 rooms
(15 facts per room).  Each room belongs to one of two namespaces:

  namespace=engineering   rooms: auth, billing, infra, search
  namespace=product       rooms: roadmap, analytics

For each of 30 queries (5 per room) we run two searches:

  1. Global  (no namespace/room): unscoped hybrid search over all 90 facts
  2. Scoped  (namespace + room): dual-lane routing with global companion lane

Metrics
-------
  P@K  — fraction of top-K results from the target room  (precision/noise)
  R@K  — fraction of per-query relevant facts retrieved in top-K  (recall)

A positive P@K lift (scoped − global) proves room routing reduces noise.
R@K must be maintained (≥ global) to prove the companion lane preserves recall.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from workflows_bench_common import (
    FastEmbedEncoder,
    add_db_args,
    connect_knowledge_backend,
    db_config_from_args,
    purge_benchmark_sources,
)

from workflows_mcp.engine.knowledge.constants import Authority, LifecycleState
from workflows_mcp.engine.knowledge.search import room_scoped_search

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
# 6 rooms × 15 facts = 90 total.  Facts are written so vocabulary overlaps
# across rooms (tokens, services, execution, data) to create semantic noise
# for the global search — making precision lift from room routing measurable.

CORPUS: list[dict[str, str]] = [
    # ── engineering / auth ──────────────────────────────────────────────────
    {
        "id": "auth-00",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "JWT access tokens are valid for 24 hours; refresh tokens are valid "
            "for 30 days and stored as HttpOnly cookies to prevent XSS theft."
        ),
    },
    {
        "id": "auth-01",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "The /auth/token endpoint accepts client_credentials and "
            "authorization_code OAuth2 grant types. Implicit flow is disabled."
        ),
    },
    {
        "id": "auth-02",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "Multi-factor authentication (MFA) is enforced for all admin "
            "accounts using TOTP via an authenticator app. Backup codes are "
            "one-time-use and hashed with bcrypt."
        ),
    },
    {
        "id": "auth-03",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "Service-to-service authentication uses mTLS certificates issued "
            "by Vault PKI. JWT bearer tokens are reserved for user sessions."
        ),
    },
    {
        "id": "auth-04",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "Password hashing uses bcrypt with cost factor 12. Plaintext "
            "passwords are never stored, logged, or transmitted after the "
            "initial TLS-encrypted submission."
        ),
    },
    {
        "id": "auth-05",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "The token introspection endpoint validates tokens issued by "
            "third-party OAuth2 providers including Google and GitHub. "
            "Validation result is cached for 60 seconds per token."
        ),
    },
    {
        "id": "auth-06",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "Session tokens are rotated on every successful authentication "
            "event. The concurrent session limit per user is 5 active tokens."
        ),
    },
    {
        "id": "auth-07",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "SSO integration uses SAML 2.0 for enterprise customers. The IdP "
            "metadata XML URL must be configured in the organisation settings "
            "before users can log in via SSO."
        ),
    },
    {
        "id": "auth-08",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "Rate limiting on /auth/login: 5 failed attempts per IP per minute "
            "triggers a 15-minute account lockout. The lockout is recorded in "
            "the audit log with the originating IP."
        ),
    },
    {
        "id": "auth-09",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "The auth service maintains a token revocation list (TRL) in Redis. "
            "Every API request checks the TRL before processing."
        ),
    },
    {
        "id": "auth-10",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "OAuth2 scopes are hierarchical: read < write < admin. Downgrading "
            "scope requires re-authentication; upgrading requires explicit user consent."
        ),
    },
    {
        "id": "auth-11",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "JWT claims include sub (user_id), org_id, roles, exp, iat, jti. "
            "The org_id claim is extracted by the API gateway and used for "
            "tenant routing — it is never accepted from request headers."
        ),
    },
    {
        "id": "auth-12",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "Cross-origin requests require a CSRF token in X-CSRF-Token header. "
            "CSRF tokens are bound to session tokens and rotate with each login."
        ),
    },
    {
        "id": "auth-13",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "The auth service exposes a JWKS endpoint at /.well-known/jwks.json "
            "for public key distribution used by downstream services to verify "
            "RS256 signatures."
        ),
    },
    {
        "id": "auth-14",
        "ns": "engineering",
        "room": "auth",
        "content": (
            "Refresh token rotation: each use of a refresh token immediately "
            "invalidates it and issues a new one. Reuse of a revoked refresh "
            "token invalidates the entire token family."
        ),
    },
    # ── engineering / billing ────────────────────────────────────────────────
    {
        "id": "billing-00",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Stripe webhooks arrive at /webhooks/stripe. Each event is "
            "idempotency-keyed by the Stripe event ID to prevent double-processing."
        ),
    },
    {
        "id": "billing-01",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Subscription invoices are generated on the 1st of each month. "
            "Failed payments trigger 3 automatic retry attempts over 7 days "
            "before the subscription is suspended."
        ),
    },
    {
        "id": "billing-02",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "The billing service issues short-lived metering tokens to "
            "execution-worker for reporting per-workflow CPU and memory usage."
        ),
    },
    {
        "id": "billing-03",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Metered billing aggregates CPU-seconds and memory-GiB-seconds per "
            "workflow execution, rolled up hourly by the billing service."
        ),
    },
    {
        "id": "billing-04",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Mid-cycle plan upgrades use Stripe's prorate_immediately strategy. "
            "Downgrades take effect at the end of the current billing period."
        ),
    },
    {
        "id": "billing-05",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Raw card data never touches our servers. Stripe.js tokenises the "
            "card client-side and submits only a Stripe payment method token."
        ),
    },
    {
        "id": "billing-06",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Refund policy: unused credits refunded within 7 days on request. "
            "Annual subscriptions carry a 30-day money-back guarantee."
        ),
    },
    {
        "id": "billing-07",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "The /billing/usage API returns aggregated execution and storage "
            "metrics per organisation per calendar month."
        ),
    },
    {
        "id": "billing-08",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Free tier limits: 1 000 workflow executions per month and 10 GB "
            "storage. Exceeding either limit suspends new execution starts "
            "until the next billing cycle."
        ),
    },
    {
        "id": "billing-09",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Tax calculation uses the Stripe Tax API. VAT and GST rates are "
            "determined from the customer's billing address at invoice creation time."
        ),
    },
    {
        "id": "billing-10",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Dunning management: subscriptions more than 14 days overdue are "
            "suspended. Service is automatically restored on successful payment."
        ),
    },
    {
        "id": "billing-11",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Enterprise accounts use purchase orders and NET-30 invoicing. "
            "Stripe is bypassed; invoices are generated and collected manually."
        ),
    },
    {
        "id": "billing-12",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "The billing metering token endpoint issues tokens with a 5-minute "
            "TTL used by execution-worker to authenticate usage reporting calls."
        ),
    },
    {
        "id": "billing-13",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Plan tiers: Starter ($49/mo), Growth ($199/mo), Enterprise (custom). "
            "Each tier unlocks additional concurrent workflow executions and "
            "extended storage."
        ),
    },
    {
        "id": "billing-14",
        "ns": "engineering",
        "room": "billing",
        "content": (
            "Billing records are retained for 7 years per financial regulations. "
            "Personal data is anonymised 90 days after contract termination."
        ),
    },
    # ── engineering / infra ──────────────────────────────────────────────────
    {
        "id": "infra-00",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Kubernetes runs on OrbStack locally and Amazon EKS in production. "
            "Each tenant organisation gets a dedicated Kubernetes namespace "
            "for hard runtime isolation."
        ),
    },
    {
        "id": "infra-01",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "The service mesh uses Istio in STRICT mTLS mode. All inter-service "
            "traffic is encrypted and mutually authenticated with auto-rotated certs."
        ),
    },
    {
        "id": "infra-02",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Container images are built with Tilt and pushed to Amazon ECR. "
            "Image tags are the Git commit SHA for reproducibility."
        ),
    },
    {
        "id": "infra-03",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Horizontal Pod Autoscaler (HPA) scales execution-worker pods on "
            "CPU utilisation. Target: 70% CPU. Min replicas: 1, max: 50."
        ),
    },
    {
        "id": "infra-04",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Node autoscaling uses Karpenter. Spot instances are used for "
            "execution-worker; on-demand instances for control-plane services."
        ),
    },
    {
        "id": "infra-05",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Deployments use rolling update strategy: max-surge=1, "
            "max-unavailable=0. A readiness probe must pass before traffic is "
            "routed to new pods."
        ),
    },
    {
        "id": "infra-06",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "NetworkPolicies restrict inter-service communication to declared "
            "callers only. Default deny-all ingress with explicit allow rules "
            "per service."
        ),
    },
    {
        "id": "infra-07",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Secrets are injected by External Secrets Operator from Doppler. "
            "Kubernetes Secret objects are ephemeral and never committed to Git."
        ),
    },
    {
        "id": "infra-08",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "PostgreSQL runs as Amazon RDS Multi-AZ with automated backups. "
            "Connection pooling is handled by a PgBouncer sidecar per service."
        ),
    },
    {
        "id": "infra-09",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Log aggregation: Fluent Bit ships logs to OpenSearch. Retention "
            "is 30 days in hot storage, 12 months in cold (S3 Glacier)."
        ),
    },
    {
        "id": "infra-10",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Distributed tracing uses OpenTelemetry with a Jaeger backend. "
            "Production sampling rate is 10%. All requests include a trace-id header."
        ),
    },
    {
        "id": "infra-11",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "SLIs: API p99 latency < 500 ms, error rate < 0.1%, measured over "
            "a 30-day rolling window. SLO breach triggers an on-call page."
        ),
    },
    {
        "id": "infra-12",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Disaster recovery targets: RTO 4 hours, RPO 1 hour. Database "
            "backups are automated daily and replicated to a second AWS region."
        ),
    },
    {
        "id": "infra-13",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "API Gateway rate limits: 100 requests/min per unauthenticated IP, "
            "1 000 requests/min per authenticated user. Exceeded requests "
            "receive HTTP 429."
        ),
    },
    {
        "id": "infra-14",
        "ns": "engineering",
        "room": "infra",
        "content": (
            "Ingress controller is AWS ALB with WAF rules. Request body size "
            "limit: 16 MB. Idle connection timeout: 60 seconds."
        ),
    },
    # ── engineering / search ─────────────────────────────────────────────────
    {
        "id": "search-00",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Memory fact embeddings use sentence-transformers/"
            "all-MiniLM-L6-v2 (384 dimensions) by default. The model name "
            "is stored per-memory so multi-model corpora are supported."
        ),
    },
    {
        "id": "search-01",
        "ns": "engineering",
        "room": "search",
        "content": (
            "pgvector stores embeddings in VECTOR(384) columns. The HNSW "
            "index is configured with m=16, ef_construction=64 for high "
            "recall at query-time ef=100."
        ),
    },
    {
        "id": "search-02",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Hybrid search fuses pgvector cosine similarity and PostgreSQL "
            "full-text search via Reciprocal Rank Fusion (RRF). "
            "Default k=60 in the RRF formula."
        ),
    },
    {
        "id": "search-03",
        "ns": "engineering",
        "room": "search",
        "content": (
            "RRF score formula: Σ(weight_i / (k + rank_i)). Both vector and "
            "FTS lanes use equal weights of 1.0. Higher rank → lower "
            "denominator → higher contribution."
        ),
    },
    {
        "id": "search-04",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Room-scoped retrieval runs two lanes concurrently: a room-filtered "
            "lane and a global companion lane capped at 20 candidates. "
            "Results are merged via a two-pass RRF."
        ),
    },
    {
        "id": "search-05",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Full-text search uses plainto_tsquery with English language "
            "configuration. The search_vector column is a tsvector updated "
            "via a trigger on content insert or update."
        ),
    },
    {
        "id": "search-07",
        "ns": "engineering",
        "room": "search",
        "content": (
            "The memory facts table has a GIN index on search_vector for fast "
            "full-text lookups. The HNSW vector index supports approximate "
            "nearest-neighbour queries."
        ),
    },
    {
        "id": "search-08",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Pre-filter candidate limit is min(limit × 10, 200) for both "
            "vector and FTS lanes before RRF re-ranking."
        ),
    },
    {
        "id": "search-09",
        "ns": "engineering",
        "room": "search",
        "content": (
            "MMR (Maximal Marginal Relevance) re-ranking is applied in "
            "query_memory strategy=context to maximise diversity of the assembled context. "
            "Embeddings are returned from search only for MMR."
        ),
    },
    {
        "id": "search-10",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Category filters use an EXISTS subquery on the junction table to "
            "avoid row duplication when a memory belongs to multiple categories."
        ),
    },
    {
        "id": "search-11",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Partial indexes on namespace and room columns enable efficient "
            "room-scoped queries: WHERE namespace = $1 AND room = $2 hits a "
            "dedicated index rather than a full scan."
        ),
    },
    {
        "id": "search-12",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Cosine similarity is returned as: 1 - (embedding <=> query_vec). "
            "Range is 0 (orthogonal vectors) to 1 (identical vectors)."
        ),
    },
    {
        "id": "search-13",
        "ns": "engineering",
        "room": "search",
        "content": (
            "Lifecycle state (ACTIVE / QUARANTINED / ARCHIVED) is applied as a "
            "WHERE clause before vector search so archived memories never "
            "appear in results."
        ),
    },
    {
        "id": "search-14",
        "ns": "engineering",
        "room": "search",
        "content": (
            "The retrieval_count column is incremented per memory on each "
            "successful retrieval for usage analytics and potential future "
            "popularity-based re-ranking."
        ),
    },
    # ── product / roadmap ────────────────────────────────────────────────────
    {
        "id": "roadmap-00",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Q2 2026 milestone: launch webhook trigger blocks for workflows. "
            "Target ship date: 30 April 2026."
        ),
    },
    {
        "id": "roadmap-01",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "The v2.0 release introduces multi-step approval workflows with "
            "pause/resume and email notifications. GA target is end of Q2."
        ),
    },
    {
        "id": "roadmap-02",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Feature backlog is managed in Linear. Sprint planning runs every "
            "two weeks. The backlog is groomed weekly by the product team."
        ),
    },
    {
        "id": "roadmap-03",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Corridor-level topology scoping (namespace/room/corridor) is "
            "planned for Q3 2026 as the next phase of the memory palace RFC."
        ),
    },
    {
        "id": "roadmap-04",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "The analytics dashboard feature is deprioritised from Q2 2026 "
            "in favour of execution monitoring improvements."
        ),
    },
    {
        "id": "roadmap-05",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Customer feature requests are scored with RICE "
            "(Reach × Impact × Confidence / Effort) before entering the sprint backlog."
        ),
    },
    {
        "id": "roadmap-06",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "The public API v2 release is gated on a full security review. "
            "Estimated completion: end of Q2 2026."
        ),
    },
    {
        "id": "roadmap-07",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Mobile app development is scheduled for H2 2026. The team will "
            "use React Native for cross-platform iOS and Android support."
        ),
    },
    {
        "id": "roadmap-08",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Roadmap items labelled enterprise require sign-off from the "
            "enterprise advisory board before being scheduled in a sprint."
        ),
    },
    {
        "id": "roadmap-09",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Execution monitoring will track per-run CPU and memory usage at "
            "1-minute granularity. This feeds into both the analytics dashboard "
            "and the billing metering pipeline."
        ),
    },
    {
        "id": "roadmap-10",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Integrations roadmap: Slack notifications (Q2 2026), GitHub "
            "Actions triggers (Q2 2026), Jira ticket creation (Q3 2026)."
        ),
    },
    {
        "id": "roadmap-11",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Self-hosted deployment (Kubernetes Helm chart) is a Q4 2026 "
            "initiative. Initial target is single-node OrbStack installations."
        ),
    },
    {
        "id": "roadmap-12",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "Each sprint ships a written async demo to stakeholders. Demos are "
            "recorded and archived in Notion with timestamp and sprint number."
        ),
    },
    {
        "id": "roadmap-13",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "The design-partner beta programme has 10 early-access customers "
            "who provide structured feedback before public GA of new features."
        ),
    },
    {
        "id": "roadmap-14",
        "ns": "product",
        "room": "roadmap",
        "content": (
            "The billing metering token refresh feature was moved from Q3 to "
            "Q2 2026 following feedback from three enterprise customers."
        ),
    },
    # ── product / analytics ──────────────────────────────────────────────────
    {
        "id": "analytics-00",
        "ns": "product",
        "room": "analytics",
        "content": (
            "User retention is measured as 30-day and 90-day rolling cohorts. "
            "Target: ≥ 60% 30-day retention for paid accounts."
        ),
    },
    {
        "id": "analytics-01",
        "ns": "product",
        "room": "analytics",
        "content": (
            "Event tracking uses Segment CDP with two destinations: Amplitude "
            "(product analytics) and Snowflake (data warehouse for ad-hoc analysis)."
        ),
    },
    {
        "id": "analytics-02",
        "ns": "product",
        "room": "analytics",
        "content": (
            "The activation metric is: user creates and successfully runs their "
            "first workflow within 7 days of signup."
        ),
    },
    {
        "id": "analytics-03",
        "ns": "product",
        "room": "analytics",
        "content": (
            "Funnel: signup → trial_start → first_workflow_run → paid_subscription. "
            "Drop-off is highest at first_workflow_run (≈ 40% of trial starts "
            "do not complete this step)."
        ),
    },
    {
        "id": "analytics-04",
        "ns": "product",
        "room": "analytics",
        "content": (
            "Dashboard data refreshes every 4 hours for standard metrics. "
            "Real-time dashboards (< 1 min lag) are available on Growth and "
            "Enterprise plans."
        ),
    },
    {
        "id": "analytics-05",
        "ns": "product",
        "room": "analytics",
        "content": (
            "Churn is defined as no successful workflow execution in 30 days "
            "for a paid account. Monthly churn rate target: < 3%."
        ),
    },
    {
        "id": "analytics-06",
        "ns": "product",
        "room": "analytics",
        "content": (
            "North Star Metric (NSM): total monthly active workflow executions "
            "across all paid organisations."
        ),
    },
    {
        "id": "analytics-07",
        "ns": "product",
        "room": "analytics",
        "content": (
            "A/B tests use a custom framework built on Unleash feature flags. "
            "Minimum sample size before reading results: 1 000 per variant."
        ),
    },
    {
        "id": "analytics-08",
        "ns": "product",
        "room": "analytics",
        "content": (
            "The analytics service consumes hourly usage metering tokens from "
            "the billing service to populate execution frequency dashboards."
        ),
    },
    {
        "id": "analytics-09",
        "ns": "product",
        "room": "analytics",
        "content": (
            "Data quality checks run daily. Anomaly detection flags any metric "
            "that deviates more than 2 standard deviations from its 30-day "
            "rolling average."
        ),
    },
    {
        "id": "analytics-10",
        "ns": "product",
        "room": "analytics",
        "content": (
            "GDPR compliance: user-identifiable analytics data is anonymised "
            "after 90 days. Personal identifiers are pseudonymised at ingestion."
        ),
    },
    {
        "id": "analytics-11",
        "ns": "product",
        "room": "analytics",
        "content": (
            "Customer health score is a composite of: execution frequency, "
            "active user count, open support tickets, and payment status."
        ),
    },
    {
        "id": "analytics-12",
        "ns": "product",
        "room": "analytics",
        "content": (
            "The weekly business review (WBR) dashboard is auto-generated from "
            "Snowflake views and distributed to leadership every Monday morning."
        ),
    },
    {
        "id": "analytics-13",
        "ns": "product",
        "room": "analytics",
        "content": (
            "Feature adoption rate is tracked per-cohort. A feature is "
            "considered 'adopted' when used by more than 20% of eligible accounts."
        ),
    },
    {
        "id": "analytics-14",
        "ns": "product",
        "room": "analytics",
        "content": (
            "The execution success rate KPI is: successful_runs / total_runs "
            "over a 7-day rolling window. Target: > 95%."
        ),
    },
]

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
# Each query specifies the target namespace+room and a list of corpus IDs
# that are the "ground-truth" relevant facts for that query.
# P@K uses target room membership; R@K uses these specific IDs.

QUERIES: list[dict[str, Any]] = [
    # ── auth ─────────────────────────────────────────────────────────────────
    {
        "query": "How long are JWT tokens valid and how is token refresh handled?",
        "ns": "engineering",
        "room": "auth",
        "relevant": ["auth-00", "auth-06", "auth-14"],
    },
    {
        "query": "What OAuth2 grant types are supported for authentication?",
        "ns": "engineering",
        "room": "auth",
        "relevant": ["auth-01", "auth-05", "auth-10"],
    },
    {
        "query": "How does service-to-service authentication work?",
        "ns": "engineering",
        "room": "auth",
        "relevant": ["auth-03", "auth-13"],
    },
    {
        "query": "How is MFA configured and enforced for admin users?",
        "ns": "engineering",
        "room": "auth",
        "relevant": ["auth-02"],
    },
    {
        "query": "What happens when login rate limits are exceeded?",
        "ns": "engineering",
        "room": "auth",
        "relevant": ["auth-08", "auth-09"],
    },
    # ── billing ───────────────────────────────────────────────────────────────
    {
        "query": "How are Stripe webhooks processed for payment events?",
        "ns": "engineering",
        "room": "billing",
        "relevant": ["billing-00"],
    },
    {
        "query": "How is per-workflow execution cost tracked and billed?",
        "ns": "engineering",
        "room": "billing",
        "relevant": ["billing-02", "billing-03"],
    },
    {
        "query": "What is the refund policy for subscriptions?",
        "ns": "engineering",
        "room": "billing",
        "relevant": ["billing-06"],
    },
    {
        "query": "How does dunning work for overdue subscription payments?",
        "ns": "engineering",
        "room": "billing",
        "relevant": ["billing-10", "billing-01"],
    },
    {
        "query": "What are the free tier execution and storage limits?",
        "ns": "engineering",
        "room": "billing",
        "relevant": ["billing-08"],
    },
    # ── infra ─────────────────────────────────────────────────────────────────
    {
        "query": "How is Kubernetes organised for multi-tenant isolation?",
        "ns": "engineering",
        "room": "infra",
        "relevant": ["infra-00"],
    },
    {
        "query": "How are secrets managed and injected into production services?",
        "ns": "engineering",
        "room": "infra",
        "relevant": ["infra-07"],
    },
    {
        "query": "What autoscaling strategy is used for execution worker pods?",
        "ns": "engineering",
        "room": "infra",
        "relevant": ["infra-03", "infra-04"],
    },
    {
        "query": "How is inter-service traffic secured in the service mesh?",
        "ns": "engineering",
        "room": "infra",
        "relevant": ["infra-01", "infra-06"],
    },
    {
        "query": "What are the disaster recovery RTO and RPO targets?",
        "ns": "engineering",
        "room": "infra",
        "relevant": ["infra-12"],
    },
    # ── search ────────────────────────────────────────────────────────────────
    {
        "query": "What embedding model is used for memory facts?",
        "ns": "engineering",
        "room": "search",
        "relevant": ["search-00", "search-06"],
    },
    {
        "query": "How does hybrid search combine vector and full-text results?",
        "ns": "engineering",
        "room": "search",
        "relevant": ["search-02", "search-03"],
    },
    {
        "query": "How does room-scoped search work with a global companion lane?",
        "ns": "engineering",
        "room": "search",
        "relevant": ["search-04", "search-11"],
    },
    {
        "query": "What index types are used for vector and full-text search?",
        "ns": "engineering",
        "room": "search",
        "relevant": ["search-01", "search-07"],
    },
    {
        "query": "How many candidate documents are pre-fetched before RRF reranking?",
        "ns": "engineering",
        "room": "search",
        "relevant": ["search-08"],
    },
    # ── roadmap ───────────────────────────────────────────────────────────────
    {
        "query": "What are the key Q2 2026 product deliverables?",
        "ns": "product",
        "room": "roadmap",
        "relevant": ["roadmap-00", "roadmap-01", "roadmap-06", "roadmap-10"],
    },
    {
        "query": "How is the feature backlog prioritised and scored?",
        "ns": "product",
        "room": "roadmap",
        "relevant": ["roadmap-02", "roadmap-05"],
    },
    {
        "query": "What is the status of the analytics dashboard feature?",
        "ns": "product",
        "room": "roadmap",
        "relevant": ["roadmap-04"],
    },
    {
        "query": "When is the self-hosted deployment option planned?",
        "ns": "product",
        "room": "roadmap",
        "relevant": ["roadmap-11"],
    },
    {
        "query": "What is the corridor-level topology scoping feature?",
        "ns": "product",
        "room": "roadmap",
        "relevant": ["roadmap-03"],
    },
    # ── analytics ─────────────────────────────────────────────────────────────
    {
        "query": "How is user retention measured and what is the target?",
        "ns": "product",
        "room": "analytics",
        "relevant": ["analytics-00"],
    },
    {
        "query": "What is the North Star Metric for the product?",
        "ns": "product",
        "room": "analytics",
        "relevant": ["analytics-06"],
    },
    {
        "query": "How is customer churn defined and what is the target rate?",
        "ns": "product",
        "room": "analytics",
        "relevant": ["analytics-05"],
    },
    {
        "query": "How does the user activation funnel work?",
        "ns": "product",
        "room": "analytics",
        "relevant": ["analytics-02", "analytics-03"],
    },
    {
        "query": "How is the customer health score calculated?",
        "ns": "product",
        "room": "analytics",
        "relevant": ["analytics-11"],
    },
]

SYSTEM_USER_UUID = uuid.UUID("00000000-0000-0000-0000-000000000001")
SOURCE_PREFIX = "bench-topo"
SOURCE_NAME = f"{SOURCE_PREFIX}:corpus-v1"


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------


async def purge_and_ingest(
    backend: Any,
    corpus: list[dict[str, str]],
    embeddings: list[list[float]],
    embedding_model: str,
) -> dict[str, str]:
    """Insert corpus facts with namespace/room set. Returns memory_id-to-corpus_id map."""
    await purge_benchmark_sources(backend, SOURCE_PREFIX)

    # Upsert source
    source_id = await _upsert_source(backend, SOURCE_NAME, "benchmark")

    item_rows: list[tuple[Any, ...]] = []
    prop_rows: list[tuple[Any, ...]] = []
    prop_id_map: dict[str, str] = {}  # memory_id -> corpus_id

    for pos, (fact, vec) in enumerate(zip(corpus, embeddings, strict=True)):
        item_id = str(uuid.uuid4())
        prop_id = str(uuid.uuid4())
        item_rows.append((item_id, source_id, f"{pos}:{fact['id']}", fact["id"]))
        prop_rows.append(
            (
                prop_id,
                item_id,
                fact["content"],
                str(vec),
                Authority.EXTRACTED,
                LifecycleState.ACTIVE,
                1.0,
                embedding_model,
                "{}",
                str(SYSTEM_USER_UUID),
                "SYSTEM",
                SOURCE_NAME,
                "benchmark",
                fact["ns"],  # namespace
                fact["room"],  # room
                None,  # corridor (not used in this benchmark)
            )
        )
        prop_id_map[prop_id] = fact["id"]

    await backend.execute_many(
        "INSERT INTO knowledge_items (id, source_id, path, title) "
        "VALUES ($1::uuid, $2::uuid, $3, $4)",
        item_rows,
    )
    await backend.execute_many(
        """
        INSERT INTO knowledge_memories
            (id, item_id, content, embedding, search_vector,
             authority, lifecycle_state, confidence,
             embedding_model, metadata,
             created_by, auth_method, source_name, source_type,
             namespace, room, corridor)
        VALUES
            ($1::uuid, $2::uuid, $3, $4::vector,
             to_tsvector('english', $3),
             $5, $6, $7, $8, $9::jsonb,
             $10::uuid, $11, $12, $13,
             $14, $15, $16)
        """,
        prop_rows,
    )
    return prop_id_map


async def _upsert_source(backend: Any, name: str, source_type: str) -> str:
    result = await backend.query(
        """
        INSERT INTO knowledge_sources (id, name, source_type, category_ids)
        VALUES ($1::uuid, $2, $3, '{}'::uuid[])
        ON CONFLICT (name) DO UPDATE
            SET source_type = EXCLUDED.source_type,
                updated_at = NOW()
        RETURNING id
        """,
        (str(uuid.uuid4()), name, source_type),
    )
    return str(result.rows[0]["id"])


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


async def run_global_search(
    backend: Any,
    query_embedding: list[float],
    query_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Unscoped hybrid search over the entire bench corpus source."""
    return await room_scoped_search(
        query_embedding,
        query_text,
        backend,
        namespace=None,
        room=None,
        source=SOURCE_NAME,
        limit=limit,
    )


async def run_room_search(
    backend: Any,
    query_embedding: list[float],
    query_text: str,
    namespace: str,
    room: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Room-scoped dual-lane search."""
    return await room_scoped_search(
        query_embedding,
        query_text,
        backend,
        namespace=namespace,
        room=room,
        source=SOURCE_NAME,
        limit=limit,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_at_k(
    results: list[dict[str, Any]],
    prop_to_corpus: dict[str, str],
    room_corpus_ids: set[str],
    k: int,
) -> float:
    """Fraction of top-K results whose corpus fact belongs to the target room."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for row in top_k if prop_to_corpus.get(str(row["id"]), "") in room_corpus_ids)
    return hits / k


def recall_at_k(
    results: list[dict[str, Any]],
    prop_to_corpus: dict[str, str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of known-relevant facts that appear in the top-K results."""
    if not relevant_ids:
        return 0.0
    top_k_corpus = {prop_to_corpus.get(str(row["id"]), "") for row in results[:k]}
    hits = sum(1 for rid in relevant_ids if rid in top_k_corpus)
    return hits / len(relevant_ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark(args: argparse.Namespace) -> None:
    encoder = FastEmbedEncoder(model_name=args.embed_model)
    db_config = db_config_from_args(args)
    backend = await connect_knowledge_backend(db_config)

    print("=" * 70)
    print("Memory Palace — Topology Routing Benchmark")
    print("=" * 70)
    print(f"Corpus:      {len(CORPUS)} facts across 6 rooms (15 per room)")
    print(f"Queries:     {len(QUERIES)} queries (5 per room)")
    print(f"Embed model: {encoder.requested_model_name} -> {encoder.model_name}")
    print(f"Limits:      P@{args.precision_k}, R@{args.recall_k}")
    print("-" * 70)

    # Encode corpus
    print("Encoding corpus... ", end="", flush=True)
    corpus_texts = [f["content"] for f in CORPUS]
    corpus_embeddings = encoder.encode(corpus_texts)
    print(f"done ({len(corpus_embeddings)} vectors)")

    # Ingest
    print("Ingesting into PostgreSQL... ", end="", flush=True)
    prop_to_corpus = await purge_and_ingest(backend, CORPUS, corpus_embeddings, encoder.model_name)
    print(f"done ({len(prop_to_corpus)} memory facts)")

    # room-key → set of corpus IDs
    room_to_corpus_ids: dict[str, set[str]] = {}
    for f in CORPUS:
        key = f"{f['ns']}/{f['room']}"
        room_to_corpus_ids.setdefault(key, set()).add(f["id"])

    # Encode queries
    query_texts = [q["query"] for q in QUERIES]
    query_embeddings = encoder.encode(query_texts)

    # ── Per-room accumulators ─────────────────────────────────────────────────
    room_keys = sorted(room_to_corpus_ids.keys())
    stats: dict[str, dict[str, list[float]]] = {
        rk: {
            "global_p": [],
            "room_p": [],
            "global_r": [],
            "room_r": [],
        }
        for rk in room_keys
    }

    pk = args.precision_k
    rk = args.recall_k

    started_at = time.perf_counter()

    print(
        f"\n{'Query':<52} {'G P@' + str(pk):<8} {'S P@' + str(pk):<8} "
        f"{'G R@' + str(rk):<8} {'S R@' + str(rk):<8} {'P lift':<8}"
    )
    print("-" * 70)

    for q, q_emb in zip(QUERIES, query_embeddings, strict=True):
        target_ns = q["ns"]
        target_room = q["room"]
        target_key = f"{target_ns}/{target_room}"
        room_ids = room_to_corpus_ids[target_key]
        relevant_ids = set(q["relevant"])

        global_results, room_results = await asyncio.gather(
            run_global_search(backend, q_emb, q["query"], limit=max(pk, rk) + 5),
            run_room_search(
                backend, q_emb, q["query"], target_ns, target_room, limit=max(pk, rk) + 5
            ),
        )

        gp = precision_at_k(global_results, prop_to_corpus, room_ids, pk)
        sp = precision_at_k(room_results, prop_to_corpus, room_ids, pk)
        gr = recall_at_k(global_results, prop_to_corpus, relevant_ids, rk)
        sr = recall_at_k(room_results, prop_to_corpus, relevant_ids, rk)
        lift = (sp - gp) if gp < 1.0 else 0.0

        stats[target_key]["global_p"].append(gp)
        stats[target_key]["room_p"].append(sp)
        stats[target_key]["global_r"].append(gr)
        stats[target_key]["room_r"].append(sr)

        label = q["query"][:50]
        sign = "+" if lift >= 0 else ""
        print(f"  {label:<50} {gp:<8.3f} {sp:<8.3f} {gr:<8.3f} {sr:<8.3f} {sign}{lift:.3f}")

    elapsed = time.perf_counter() - started_at

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-ROOM SUMMARY")
    print("=" * 70)
    print(
        f"{'Room':<26} {'G P@' + str(pk):<8} {'S P@' + str(pk):<8} "
        f"{'P lift':<8} {'G R@' + str(rk):<8} {'S R@' + str(rk):<8} {'R delta':<8}"
    )
    print("-" * 70)

    all_gp: list[float] = []
    all_sp: list[float] = []
    all_gr: list[float] = []
    all_sr: list[float] = []

    per_room_summary: list[dict[str, Any]] = []

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    for rk_key in room_keys:
        s = stats[rk_key]
        gp_avg = avg(s["global_p"])
        sp_avg = avg(s["room_p"])
        gr_avg = avg(s["global_r"])
        sr_avg = avg(s["room_r"])
        p_lift = sp_avg - gp_avg
        r_delta = sr_avg - gr_avg
        sign_p = "+" if p_lift >= 0 else ""
        sign_r = "+" if r_delta >= 0 else ""
        print(
            f"  {rk_key:<24} {gp_avg:<8.3f} {sp_avg:<8.3f} "
            f"{sign_p}{p_lift:<7.3f} {gr_avg:<8.3f} {sr_avg:<8.3f} "
            f"{sign_r}{r_delta:.3f}"
        )
        all_gp.extend(s["global_p"])
        all_sp.extend(s["room_p"])
        all_gr.extend(s["global_r"])
        all_sr.extend(s["room_r"])
        per_room_summary.append(
            {
                "room": rk_key,
                f"global_p@{pk}": round(gp_avg, 4),
                f"scoped_p@{pk}": round(sp_avg, 4),
                "p_lift": round(p_lift, 4),
                f"global_r@{rk}": round(gr_avg, 4),
                f"scoped_r@{rk}": round(sr_avg, 4),
                "r_delta": round(r_delta, 4),
            }
        )

    overall_gp = avg(all_gp)
    overall_sp = avg(all_sp)
    overall_gr = avg(all_gr)
    overall_sr = avg(all_sr)
    overall_p_lift = overall_sp - overall_gp
    overall_r_delta = overall_sr - overall_gr
    lift_factor = (overall_sp / overall_gp) if overall_gp > 0 else float("inf")

    print("-" * 70)
    sign_p = "+" if overall_p_lift >= 0 else ""
    sign_r = "+" if overall_r_delta >= 0 else ""
    print(
        f"  {'OVERALL':<24} {overall_gp:<8.3f} {overall_sp:<8.3f} "
        f"{sign_p}{overall_p_lift:<7.3f} {overall_gr:<8.3f} {overall_sr:<8.3f} "
        f"{sign_r}{overall_r_delta:.3f}"
    )

    print()
    print(
        f"Precision lift: {sign_p}{overall_p_lift:.3f}  "
        f"({lift_factor:.2f}x)   Recall delta: {sign_r}{overall_r_delta:.3f}"
    )
    print(f"Elapsed: {elapsed:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_slug = encoder.model_name.replace("/", "-").replace(".", "_")
    out_path = Path(__file__).parent / f"results_workflows_topology_{model_slug}_{ts}.json"
    result_doc = {
        "benchmark": "topology_routing",
        "timestamp": ts,
        "embed_model": encoder.model_name,
        "corpus_size": len(CORPUS),
        "query_count": len(QUERIES),
        "rooms": len(room_keys),
        f"overall_global_p@{pk}": round(overall_gp, 4),
        f"overall_scoped_p@{pk}": round(overall_sp, 4),
        "p_lift_absolute": round(overall_p_lift, 4),
        "p_lift_factor": round(lift_factor, 4),
        f"overall_global_r@{rk}": round(overall_gr, 4),
        f"overall_scoped_r@{rk}": round(overall_sr, 4),
        "r_delta": round(overall_r_delta, 4),
        "elapsed_seconds": round(elapsed, 1),
        "per_room": per_room_summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_doc, indent=2))
    print(f"\nResults saved to: {out_path.relative_to(Path(__file__).parent.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topology routing precision benchmark for workflows-mcp memory retrieval."
    )
    add_db_args(parser)
    parser.add_argument(
        "--embed-model",
        default="default",
        help="Embedding model alias or HuggingFace model name.",
    )
    parser.add_argument(
        "--precision-k",
        type=int,
        default=5,
        help="K for precision@K measurement (default: 5).",
    )
    parser.add_argument(
        "--recall-k",
        type=int,
        default=5,
        help="K for recall@K measurement (default: 5).",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
