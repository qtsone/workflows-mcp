# Memory operation examples (copy/paste)

## Query

```json
{
  "operation": "query",
  "scope": {
    "palace": "acme",
    "wing": "workflows",
    "room": "memory-engine",
    "compartment": "contract-r2"
  },
  "query": {
    "mode": "search",
    "text": "schema epoch",
    "radius": 1
  }
}
```

## Ingest

```json
{
  "operation": "ingest",
  "scope": {
    "palace": "acme",
    "wing": "workflows",
    "room": "memory-engine",
    "compartment": "contract-r2"
  },
  "record": {
    "format": "raw",
    "content": "Memory active",
    "memory_tier": "direct"
  }
}
```

## Maintain

```json
{
  "operation": "maintain",
  "maintenance": {"mode": "community_refresh"}
}
```

## Graph upsert (place)

```json
{
  "operation": "graph_upsert",
  "scope": {
    "palace": "acme",
    "wing": "workflows",
    "room": "memory-engine",
    "compartment": "contract-r2"
  },
  "graph": {
    "kind": "place",
    "place_name": "Memory API (current)",
    "place_type": "feature"
  }
}
```
