-- PageIndex submission cache: map content hash (SHA-256 of PDF bytes) to doc_id
-- so the same file is not re-uploaded. Run this in Supabase SQL Editor (public schema).

CREATE TABLE IF NOT EXISTS public.pageindex_submission_cache (
  content_hash TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT (NOW()),
  filename TEXT
);

-- Optional: index for listing recent entries (e.g. by created_at)
-- CREATE INDEX IF NOT EXISTS idx_pageindex_submission_cache_created_at
--   ON public.pageindex_submission_cache (created_at DESC);

COMMENT ON TABLE public.pageindex_submission_cache IS
  'Cache for PageIndex submission docs: content_hash (SHA-256 of PDF) -> doc_id to avoid re-uploading the same file.';
