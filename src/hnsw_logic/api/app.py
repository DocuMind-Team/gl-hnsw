from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi import BackgroundTasks, FastAPI, HTTPException

from hnsw_logic.api.schemas import JobRequest, SearchRequest, SearchResponseModel
from hnsw_logic.core.utils import to_jsonable
from hnsw_logic.services.bootstrap import build_app


def create_app(root_dir=None) -> FastAPI:
    container = build_app(root_dir=root_dir)
    executor = ThreadPoolExecutor(max_workers=2)
    app = FastAPI(title="gl-hnsw")

    def submit_job(job_type: str, fn):
        job = container.jobs.create(job_type, "{}")

        def _runner():
            container.jobs.update(job.job_id, "running", "started")
            try:
                fn()
                container.jobs.update(job.job_id, "completed", "done")
            except Exception as exc:  # pragma: no cover
                container.jobs.update(job.job_id, "failed", str(exc))

        executor.submit(_runner)
        return {"job_id": job.job_id, "state": job.state}

    @app.post("/api/v1/build/embeddings")
    def build_embeddings(_: JobRequest):
        return submit_job("build_embeddings", container.pipeline.build_embeddings)

    @app.post("/api/v1/build/hnsw")
    def build_hnsw(_: JobRequest):
        return submit_job("build_hnsw", container.pipeline.build_hnsw)

    @app.post("/api/v1/build/profile")
    def profile(_: JobRequest):
        return submit_job("profile_docs", container.pipeline.profile_docs)

    @app.post("/api/v1/build/discover")
    def discover(_: JobRequest):
        return submit_job("discover_edges", container.pipeline.discover_edges)

    @app.post("/api/v1/admin/revalidate")
    def revalidate(_: JobRequest):
        return submit_job("revalidate_edges", container.pipeline.revalidate_edges)

    @app.get("/api/v1/admin/jobs/{job_id}")
    def get_job(job_id: str):
        job = container.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.get("/api/v1/admin/health")
    def health():
        return {
            "status": "ok",
            "recent_jobs": [to_jsonable(job) for job in container.jobs.recent()],
            "graph_stats": container.graph_memory_store.read(),
            "doc_briefs": len(container.brief_store.all()),
        }

    @app.post("/api/v1/search", response_model=SearchResponseModel)
    def search(request: SearchRequest):
        result = container.retrieval.search(request.query, top_k=request.top_k)
        return {"query": result.query, "hits": [to_jsonable(hit) for hit in result.hits]}

    return app
