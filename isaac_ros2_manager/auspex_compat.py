"""auspex_compat - bridge the old auspex_know document-store calls to the new
instance/slot knowledge API on auspex_msgs/auspex_knowledge `main`.

Interface change (halloween-demo -> main)
-----------------------------------------
    OLD                                   NEW
    QueryKnowledge(collection)            GetAllInstances(frame)
      -> string[] answer                    -> string[] instances
    WriteKnowledge(collection, path,      UpsertSubframe(frame, instance_id,
                   entity) -> success                    subframe, item) -> success

Two things changed beyond the rename:

1. Collections are now "frames" and some were renamed. In particular the old
   "platform" collection is the "uav" frame on main. See FRAME_MAP.

2. There is no "write a whole document at a JSONPath" call any more. Identity is
   explicit (`instance_id` - the value the old JSONPath filtered on, e.g.
   team_id / area_id). `upsert_subframe` stores the entity as ONE subframe; its
   top-level keys become slots, so a read returns `{subframe: {key: value}}`.

CAVEAT: data written through this shim is NOT in the canonical shape the
KnowledgeCollector produces from typed topics (id_field + per-topic subframes).
It is self-consistent for code that both writes and reads through this module
(use `unwrap()` on reads), but will not line up with collector/fluent data.
"""
import json

from auspex_msgs.srv import GetAllInstances, UpsertSubframe

KNOW_NS = "/auspex_know"
DEFAULT_SUBFRAME = "data"

# old collection name -> new frame name. Webots/Isaac demo path keeps "platform" as-is
# so team_manager (rangers), the oracle ("_oracle_" poachers) and the webapp all share one
# frame. (Mapping platform->uav isolates the team_manager from everyone else -- the bug we
# hit on the Webots side; AMPLE/oracle/webapp all read the "platform" frame.)
FRAME_MAP = {
    "platform": "platform",
    "object": "object",
    "mission": "mission",
    "area": "area",
}


def frame_of(collection):
    """Map an old collection name to the new frame name (identity if unknown)."""
    return FRAME_MAP.get(collection, collection)


def make_query_client(node, **kwargs):
    """Replacement for create_client(QueryKnowledge, '/auspex_know/query_knowledge')."""
    return node.create_client(GetAllInstances, f"{KNOW_NS}/get_all_instances", **kwargs)


def make_write_client(node, **kwargs):
    """Replacement for create_client(WriteKnowledge, '/auspex_know/write_knowledge')."""
    return node.create_client(UpsertSubframe, f"{KNOW_NS}/upsert_subframe", **kwargs)


def query_request(collection):
    """Build a GetAllInstances request from an old QueryKnowledge collection."""
    return GetAllInstances.Request(frame=frame_of(collection))


def write_request(collection, instance_id, entity, subframe=DEFAULT_SUBFRAME):
    """Build an UpsertSubframe request from an old WriteKnowledge write.

    `instance_id` is the identity the old JSONPath filtered on. `entity` may be a
    dict or an already-serialised JSON string.
    """
    item = entity if isinstance(entity, str) else json.dumps(entity)
    return UpsertSubframe.Request(
        frame=frame_of(collection),
        instance_id=str(instance_id),
        subframe=subframe,
        item=item,
    )


def unwrap(instances, subframe=DEFAULT_SUBFRAME):
    """Recover flat entity dicts from a GetAllInstances `.instances` response.

    Each element is a JSON string shaped `{subframe: {slot: value}}`. Returns the
    inner dict for `subframe` when present, otherwise the whole instance dict.
    """
    out = []
    for raw in instances:
        try:
            doc = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(doc, dict) and subframe in doc:
            out.append(doc[subframe])
        else:
            out.append(doc)
    return out
