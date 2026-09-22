"""The durable run archive: every run kept so it can be evaluated later.

The operator's ask was "all steps, all states, all decisions need to be saved so
we can evaluate them" — so these tests pin the things that make that true: a run
survives the process that produced it, the decisions are lifted out of the feed
so an evaluation can walk them, and a store that cannot open degrades to "not
saving" rather than failing the run it was supposed to record.
"""

from __future__ import annotations

import pytest

from browser_agent.runstore import RunStore, _decisions_from
from browser_agent.tasks import Task, TaskRunner, TaskStatus, register


class _OkRecipe:
    name = "thread.ok"
    description = "succeeds"
    entry_url = ""

    async def run(self, session, payload):
        return {"ran": True}


def _register_ok() -> None:
    register(_OkRecipe())


@pytest.fixture
def runner_factory():
    """A runner whose session never opens a browser — same shape as test_thread."""
    from browser_agent.config import load_settings

    def make() -> TaskRunner:
        class _Stub:
            async def goto(self, url, **kw):
                return None

            async def page(self):
                return None

        return TaskRunner(load_settings(), _Stub(), agent_runner=None)

    return make


def _store(tmp_path, **kw) -> RunStore:
    return RunStore(tmp_path / "runs.db", profile="pytest", **kw)


def _task(**kw) -> Task:
    base = dict(
        recipe="plan.task",
        payload={"task": "find flights"},
        id="aaa111",
        status=TaskStatus.DONE,
        detail="recipe succeeded",
    )
    base.update(kw)
    return Task(**base)


# ---- what is kept ----------------------------------------------------------


def test_a_finished_run_survives_the_process(tmp_path):
    # The whole point: the record must outlive the task object, because a deploy
    # recreates the pod and every in-memory task with it.
    store = _store(tmp_path)
    task = _task(
        activity=[
            {"at": 1.0, "kind": "info", "text": "plan: 3 step(s)"},
            {"at": 2.0, "kind": "step", "text": "1/3 navigate: open it", "step": 1},
            {"at": 3.0, "kind": "gate", "text": "laya confirm: yes (conf 0.81)"},
        ],
        result={"agent_result": "found 11 flights"},
        thread_id="t1",
        attempt=2,
        used_agent=True,
    )
    assert store.save(task) is True

    # A brand-new store on the same file is a new process reading the same disk.
    reopened = _store(tmp_path)
    got = reopened.get("aaa111")
    assert got is not None
    assert got.recipe == "plan.task"
    assert got.thread_id == "t1"
    assert got.attempt == 2
    assert got.used_agent is True
    assert got.payload == {"task": "find flights"}
    assert got.result == {"agent_result": "found 11 flights"}
    assert [e["text"] for e in got.activity] == [
        "plan: 3 step(s)", "1/3 navigate: open it", "laya confirm: yes (conf 0.81)"]


def test_decisions_are_lifted_out_of_the_feed():
    # "All decisions need to be saved" is a different list from "all steps": an
    # evaluation walks what the run *chose*. A Laya verdict and an agent step
    # are decisions; a navigate is not.
    activity = [
        {"kind": "info", "text": "plan: 2 step(s)"},
        {"kind": "step", "text": "1/2 navigate"},
        {"kind": "gate", "text": "laya: no (conf 0.62)"},
        {"kind": "agent", "text": "step 3: click the search box"},
        {"kind": "error", "text": "step 2 failed"},
    ]
    got = _decisions_from(activity)
    assert [d["kind"] for d in got] == ["gate", "agent"]
    assert got[0]["text"] == "laya: no (conf 0.62)"


def test_saved_run_carries_its_decisions(tmp_path):
    store = _store(tmp_path)
    task = _task(activity=[
        {"at": 1.0, "kind": "gate", "text": "laya: yes (conf 0.90)"},
        {"at": 2.0, "kind": "step", "text": "1/1 click"},
    ])
    store.save(task)
    got = store.get("aaa111")
    assert [d["text"] for d in got.decisions] == ["laya: yes (conf 0.90)"]


def test_the_newest_are_kept_and_the_oldest_pruned(tmp_path):
    # Bounded, because the data volume is 1 GiB and an unbounded archive on it
    # would eventually fill. Count, not age: the point is to keep recent runs.
    store = _store(tmp_path, keep=3)
    for i in range(5):
        t = _task(id=f"id{i}", created_at=100.0 + i)
        store.save(t)
    kept = [r.task_id for r in store.list(limit=10)]
    assert kept == ["id4", "id3", "id2"]
    assert store.count() == 3


def test_list_can_narrow_to_one_thread(tmp_path):
    store = _store(tmp_path)
    store.save(_task(id="a", thread_id="t1", created_at=1.0))
    store.save(_task(id="b", thread_id="t2", created_at=2.0))
    only = store.list(thread_id="t2")
    assert [r.task_id for r in only] == ["b"]


# ---- degrading, not failing ------------------------------------------------


def test_an_unopenable_store_degrades_to_not_saving(tmp_path):
    # A read-only or full volume must cost the record, never the run: the store
    # is opened at import, and an exception there would crashloop the pod.
    # A *file* where the parent directory should be is the portable way to make
    # both the mkdir and the connect fail.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("in the way")
    store = RunStore(blocker / "runs.db", profile="pytest")
    assert store.enabled is False
    assert store.save(_task()) is False
    assert store.list() == []
    assert store.get("aaa111") is None
    assert store.count() == 0


def test_a_save_failure_is_swallowed(tmp_path):
    # save() is called from the runner's finally, after the task has finished.
    # Raising there would turn a done task into a crashed worker.
    # A circular payload is the honest trigger: the serializer used to write the
    # row has ``default=str``, so an un-serialisable *object* still gets stored
    # as its repr — but a cycle cannot be.
    store = _store(tmp_path)
    circular: dict = {}
    circular["self"] = circular
    broken = _task()
    broken.payload = circular
    assert store.save(broken) is False
    assert store.enabled is True  # the store itself still works


# ---- the runner writes it --------------------------------------------------


class _StubStore:
    def __init__(self) -> None:
        self.saved: list[Task] = []

    def save(self, task: Task) -> bool:
        self.saved.append(task)
        return True


def test_the_runner_archives_a_finished_attempt():
    stub = _StubStore()
    runner = TaskRunner.__new__(TaskRunner)
    runner.runs = stub

    task = _task(id="run1")
    task.activity = [{"at": 1.0, "kind": "step", "text": "did a thing"}]
    runner._archive(task)

    assert [t.id for t in stub.saved] == ["run1"]
    assert stub.saved[0].activity[0]["text"] == "did a thing"


def test_a_runner_without_a_store_still_runs():
    runner = TaskRunner.__new__(TaskRunner)
    runner.runs = None
    runner._archive(_task())  # must not raise


def test_a_task_with_no_thread_is_its_own_thread(tmp_path):
    store = _store(tmp_path)
    store.save(_task(id="solo"))
    assert store.get("solo").thread_id == "solo"


@pytest.mark.parametrize("status", ["done", "failed", "blocked"])
def test_every_terminal_status_is_recordable(tmp_path, status):
    # "All states need to be saved" — including the bad ones, which are exactly
    # the runs the operator wants to study.
    store = _store(tmp_path)
    store.save(_task(id=f"s-{status}", status=TaskStatus(status), detail=f"was {status}"))
    got = store.get(f"s-{status}")
    assert got.status == status
    assert got.detail == f"was {status}"


# ---- answering an archived run ---------------------------------------------


def test_a_record_can_become_the_next_attempt(runner_factory):
    # The reason the archive exists is to study an old run; "say what to do
    # differently" on one is what makes the study actionable, and after a deploy
    # that run exists only as a record — there is no task object to retry.
    runner = runner_factory()
    _register_ok()
    first = runner.submit("thread.ok", {"task": "one"})

    second = runner.resubmit(
        thread_id=first.thread_id,
        attempt=first.attempt,
        recipe="thread.ok",
        payload={"task": "two"},
        parent_id=first.id,
    )

    assert second.thread_id == first.thread_id
    assert second.attempt == 2
    assert second.parent_id == first.id
    assert second.payload["task"] == "two"


def test_a_resubmit_from_a_record_gets_no_brief_when_nothing_is_in_memory(runner_factory):
    # The brief comes from what is still in memory for the thread. After a
    # restart that is nothing, and an empty brief must be *absent* rather than an
    # empty string, so a prompt is never handed a dangling "Earlier attempts:".
    runner = runner_factory()
    _register_ok()
    nxt = runner.resubmit(
        thread_id="thread-from-a-previous-life",
        attempt=3,
        recipe="thread.ok",
        payload={"task": "again"},
        parent_id="gone",
    )
    assert "history" not in nxt.payload
    assert nxt.attempt == 4
