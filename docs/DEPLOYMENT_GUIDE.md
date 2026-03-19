# Deployment Guide

This guide covers deploying the Invisible Variables Engine locally using Docker Compose, as well as notes for production environments.

---

## Local Deployment with Docker Compose

### Prerequisites

- Docker Engine 24+ and Docker Compose v2+
- 4 GB RAM minimum (8 GB recommended for large datasets)
- Ports 5432, 6379, 8000, 5555, and 8501 available

### Step 1 — Clone and configure

```bash
git clone https://github.com/your-org/invisible-variables-engine.git
cd invisible-variables-engine
cp .env.example .env
```

Edit `.env` to set secure credentials before starting services. At minimum, change:

```bash
SECRET_KEY=your-secure-random-string-at-least-32-chars
POSTGRES_PASSWORD=your-secure-db-password
REDIS_PASSWORD=your-secure-redis-password
FLOWER_PASSWORD=your-secure-flower-password
VALID_API_KEYS=your-api-key-1,your-api-key-2
```

### Step 2 — Build and start

```bash
# Build and launch all 6 services
docker compose up --build -d

# Or use the Makefile shortcut
make dev
```

### Step 3 — Verify all services are healthy

```bash
# Check container status
docker compose ps

# API liveness
curl http://localhost:8000/api/v1/health

# API readiness (DB + Redis)
curl http://localhost:8000/api/v1/health/ready
```

---

## Environment Variables

### Application

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Environment name (`development`, `staging`, `production`) |
| `SECRET_KEY` | — | Application secret key (min 32 chars) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `API_PORT` | `8000` | FastAPI server port |
| `DEBUG` | `true` | Enable debug mode |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY_HEADER` | `X-API-Key` | HTTP header name for API key authentication |
| `VALID_API_KEYS` | `dev-key-1,dev-key-2` | Comma-separated list of valid API keys |

### PostgreSQL

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_HOST` | `localhost` | Database host (use `postgres` inside Docker) |
| `POSTGRES_PORT` | `5432` | Database port |
| `POSTGRES_USER` | `ive` | Database user |
| `POSTGRES_PASSWORD` | `ivepassword` | Database password |
| `POSTGRES_DB` | `ive_db` | Database name |
| `DATABASE_URL` | (assembled) | Full async connection string |
| `DATABASE_POOL_SIZE` | `10` | Connection pool size |
| `DATABASE_MAX_OVERFLOW` | `20` | Max overflow connections |

### Redis

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis host (use `redis` inside Docker) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `redispassword` | Redis password |
| `REDIS_URL` | (assembled) | Full Redis URL with password |

### Celery

| Variable | Default | Description |
|----------|---------|-------------|
| `CELERY_CONCURRENCY` | `4` | Number of worker processes |
| `CELERY_TASK_SERIALIZER` | `json` | Task serialization format |
| `CELERY_RESULT_EXPIRES` | `86400` | Result expiry in seconds (24h) |
| `CELERY_MAX_TASKS_PER_CHILD` | `100` | Max tasks before worker restart |

### Flower

| Variable | Default | Description |
|----------|---------|-------------|
| `FLOWER_PORT` | `5555` | Flower dashboard port |
| `FLOWER_USER` | `admin` | Flower basic auth username |
| `FLOWER_PASSWORD` | `flowerpass` | Flower basic auth password |

### Streamlit

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_PORT` | `8501` | Streamlit UI port |
| `API_BASE_URL` | `http://localhost:8000` | API endpoint for Streamlit to call |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `ARTIFACT_STORE_TYPE` | `local` | Storage backend (`local` or `s3`) |
| `ARTIFACT_BASE_DIR` | `/app/artifacts` | Local storage path |
| `S3_BUCKET_NAME` | `ive-artifacts` | S3 bucket (when using S3 backend) |

### ML Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFAULT_CV_FOLDS` | `5` | Default K for cross-validation |
| `RANDOM_SEED` | `42` | Global random seed for reproducibility |
| `MAX_FEATURES` | `100` | Max features to use in modeling |
| `MIN_CLUSTER_SIZE` | `10` | HDBSCAN minimum cluster size |
| `SHAP_SAMPLE_SIZE` | `500` | Max samples for SHAP computation |

---

## Service Ports

| Service | Port | URL |
|---------|------|-----|
| FastAPI API | 8000 | `http://localhost:8000` |
| Swagger Docs | 8000 | `http://localhost:8000/docs` |
| ReDoc | 8000 | `http://localhost:8000/redoc` |
| PostgreSQL | 5432 | `postgresql://ive:ivepassword@localhost:5432/ive_db` |
| Redis | 6379 | `redis://:redispassword@localhost:6379/0` |
| Flower | 5555 | `http://localhost:5555` |
| Streamlit | 8501 | `http://localhost:8501` |

---

## Common Startup Commands

```bash
# Start all services
make dev

# Start only infrastructure (DB + Redis, no app)
docker compose up postgres redis -d

# Run API locally (requires local Postgres + Redis)
make dev-local

# Run Celery worker locally
make worker-local

# Run Streamlit locally
make streamlit-local

# Run database migrations
make migrate

# Seed database with demo data
make seed

# Generate synthetic test datasets
make generate-data
```

---

## Healthcheck Endpoints

### Liveness — `GET /api/v1/health`

Returns `200` if the API process is running. Does not verify database or Redis connectivity. Used by load balancers and Docker health checks.

```json
{
  "status": "healthy",
  "service": "ive-engine",
  "version": "0.1.0",
  "timestamp": "2026-03-19T12:00:00+00:00"
}
```

### Readiness — `GET /api/v1/health/ready`

Returns `200` if PostgreSQL and Redis are both reachable. Returns `503` if any dependency is unavailable.

```json
{
  "status": "ready",
  "checks": {
    "database": "healthy",
    "redis": "healthy"
  },
  "timestamp": "2026-03-19T12:00:00+00:00"
}
```

---

## Flower Login

Navigate to `http://localhost:5555` and authenticate with:

- **Username**: value of `FLOWER_USER` (default: `admin`)
- **Password**: value of `FLOWER_PASSWORD` (default: `flowerpass`)

The Flower dashboard shows:
- Active, completed, and failed Celery tasks
- Worker resource utilization
- Task execution history and timing
- Queue depth for `default`, `analysis`, and `high_priority`

---

## Production Deployment Notes

### Security

- Rotate `SECRET_KEY`, `VALID_API_KEYS`, and all passwords before deploying.
- Use TLS termination in front of the FastAPI server (e.g., Nginx, Traefik, or cloud load balancer).
- Restrict Flower access to internal networks or VPN.
- Set `ENV=production` and `DEBUG=false`.

### Database

- Use a managed PostgreSQL service (AWS RDS, Google Cloud SQL, Azure Database) instead of the Docker container.
- Configure `DATABASE_POOL_SIZE` and `DATABASE_MAX_OVERFLOW` based on expected concurrency.
- Set up automated backups and point-in-time recovery.

### Redis

- Use a managed Redis service (AWS ElastiCache, Redis Cloud).
- Ensure `REDIS_PASSWORD` is set to a strong value and rotated periodically.

### Celery Workers

- Scale workers horizontally by deploying multiple worker containers.
- Adjust `CELERY_CONCURRENCY` based on CPU cores available per worker.
- Monitor queue depth via Flower and scale workers accordingly.
- Set `CELERY_MAX_TASKS_PER_CHILD` to prevent memory leaks from long-running ML computations.

### Storage

- For production, set `ARTIFACT_STORE_TYPE=s3` and configure AWS credentials.
- Ensure the `/app/artifacts` volume is persistent and backed up if using local storage.

### Observability

- Configure `SENTRY_DSN` for error tracking.
- Enable `ENABLE_METRICS=true` and expose Prometheus metrics on port `PROMETHEUS_PORT`.
- Forward structlog output to your log aggregation service (ELK, Datadog, etc.).

### Resource Requirements

| Service | CPU | RAM | Storage |
|---------|-----|-----|---------|
| API | 1 core | 512 MB | — |
| Worker | 2–4 cores | 2–4 GB | — |
| PostgreSQL | 1 core | 1 GB | 10 GB+ |
| Redis | 0.5 core | 256 MB | 1 GB |
| Streamlit | 0.5 core | 512 MB | — |
