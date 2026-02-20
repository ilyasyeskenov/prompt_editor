-- Store document trees built by the pageindex module (no PageIndex API).
-- Run in Supabase SQL Editor (public schema).

CREATE TABLE IF NOT EXISTS public.pageindex_trees (
  doc_id TEXT PRIMARY KEY,
  tree_json JSONB NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMPTZ DEFAULT (NOW())
);

COMMENT ON TABLE public.pageindex_trees IS
  'Document trees from pageindex module: doc_id -> tree (node_id, title, text/summary, nodes).';
