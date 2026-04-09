# CrowdShield AI

This repository contains a production-oriented microservices scaffold for real-time crowd behavior anomaly detection.

## Architecture

Pipeline:
Ingestion -> Vision -> Pose -> Behavior -> Alert -> API Gateway -> Frontend

## Project Structure

- backend/: Python microservices and shared contracts
  - ingestion-service/: Frame capture and publication to Redis streams
  - vision-service/: Person detection and tracking
  - pose-service/: Pose extraction and feature engineering
  - behavior-service/: Sequence modeling and anomaly classification
  - alert-service/: Alert decisioning, severity, and persistence
  - api-gateway/: Public REST and WebSocket interface
  - shared/: Shared contracts, stream names, messaging, and observability helpers
- frontend/: UI scaffold for dashboard and service monitoring
- training/: Data, model artifacts, configs, and training scripts

Each backend service follows clean architecture boundaries:
- app/domain: Core business entities and rules
- app/application: Use cases and orchestration logic
- app/infrastructure: Technical adapters for Redis, DB, and ML runtime
- app/interfaces: HTTP and event handlers
- app/main.py: FastAPI entrypoint with health and status endpoints

## Deployment Notes

- Dockerfiles are intentionally empty placeholders as requested.
- Add runnable service Docker instructions before building images.
- Local orchestration is defined in docker-compose.yml with Redis and PostgreSQL.

## Next Implementation Steps

1. Implement strict Redis stream consumers/producers per service contract.
2. Add robust startup/shutdown lifecycle hooks and retry policies.
3. Wire model loading pipelines for YOLO, MediaPipe, and LSTM.
4. Add unit, contract, and integration tests under each service tests/.
5. Replace frontend placeholder with a React app and WebSocket dashboard.
