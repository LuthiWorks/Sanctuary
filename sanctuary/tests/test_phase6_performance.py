"""Tests for Phase 6.5: Performance.

Tests cover:
- CognitiveProfiler: cycle profiling, phase timing, bottleneck detection
- AdaptiveCycleRate: rate adjustment, input/latency/arousal factors
- LazyEmbeddingCache: cache hits/misses, eviction, TTL
- AsyncSubsystemProcessor: parallel execution, dependency ordering, timeout
"""

import asyncio
import time

import pytest
import pytest_asyncio

from sanctuary.performance.profiler import (
    CognitiveProfiler,
    CycleProfile,
    ProfilerConfig,
)
from sanctuary.performance.adaptive_rate import (
    AdaptiveCycleRate,
    AdaptiveRateConfig,
)
from sanctuary.performance.lazy_embeddings import (
    LazyEmbeddingCache,
    LazyEmbeddingConfig,
)
from sanctuary.performance.async_processor import (
    AsyncProcessorConfig,
    AsyncSubsystemProcessor,
)


# =========================================================================
# CognitiveProfiler
# =========================================================================


class TestCognitiveProfiler:
    """Tests for cognitive cycle profiling."""

    def test_context_manager_profiling(self):
        profiler = CognitiveProfiler()
        with profiler.cycle(cycle_num=1) as p:
            with p.phase("input"):
                time.sleep(0.001)
            with p.phase("think"):
                time.sleep(0.001)
        assert len(profiler._profiles) == 1
        assert profiler._profiles[0].total_ms > 0

    def test_phase_timing_recorded(self):
        profiler = CognitiveProfiler()
        with profiler.cycle(cycle_num=1) as p:
            with p.phase("fast"):
                pass
            with p.phase("slow"):
                time.sleep(0.005)
        profile = profiler._profiles[0]
        assert "fast" in profile.phases
        assert "slow" in profile.phases
        assert profile.phases["slow"] > profile.phases["fast"]

    def test_manual_phase_recording(self):
        profiler = CognitiveProfiler()
        profiler.record_phase(cycle=1, phase="input", duration_ms=10.0)
        profiler.record_phase(cycle=1, phase="think", duration_ms=30.0)
        assert len(profiler._profiles) == 1
        assert profiler._profiles[0].phases["think"] == 30.0

    def test_summary(self):
        profiler = CognitiveProfiler()
        for i in range(5):
            profiler.record_phase(cycle=i, phase="input", duration_ms=10.0)
            profiler.record_phase(cycle=i, phase="think", duration_ms=30.0)
        summary = profiler.get_summary()
        assert summary.total_cycles == 5
        assert summary.bottleneck == "think"
        assert summary.bottleneck_pct > 0.5

    def test_slow_cycle_detection(self):
        config = ProfilerConfig(slow_cycle_threshold_ms=10.0)
        profiler = CognitiveProfiler(config=config)
        profiler.record_profile(CycleProfile(cycle=1, total_ms=50.0))
        assert len(profiler.get_slow_cycles()) == 1

    def test_phase_timeline(self):
        profiler = CognitiveProfiler()
        for i in range(5):
            profiler.record_phase(cycle=i, phase="think", duration_ms=10.0 * i)
        timeline = profiler.get_phase_timeline("think")
        assert len(timeline) == 5
        assert timeline[-1]["ms"] == 40.0

    def test_summary_empty(self):
        profiler = CognitiveProfiler()
        summary = profiler.get_summary()
        assert summary.total_cycles == 0

    def test_stats(self):
        profiler = CognitiveProfiler()
        profiler.record_phase(cycle=1, phase="x", duration_ms=5.0)
        stats = profiler.get_stats()
        assert stats["total_profiled"] == 1


# =========================================================================
# AdaptiveCycleRate
# =========================================================================


class TestAdaptiveCycleRate:
    """Tests for adaptive cycle rate."""

    def test_default_rate(self):
        rate = AdaptiveCycleRate()
        assert rate.current_rate_hz == 10.0

    def test_input_increases_rate(self):
        rate = AdaptiveCycleRate()
        delay_low = rate.compute_delay(input_queue_depth=0)
        rate2 = AdaptiveCycleRate()
        delay_high = rate2.compute_delay(input_queue_depth=10)
        assert delay_high < delay_low

    def test_high_latency_decreases_rate(self):
        rate = AdaptiveCycleRate()
        delay_fast = rate.compute_delay(last_cycle_ms=10.0)
        rate2 = AdaptiveCycleRate()
        delay_slow = rate2.compute_delay(last_cycle_ms=200.0)
        assert delay_slow > delay_fast

    def test_arousal_increases_rate(self):
        rate = AdaptiveCycleRate()
        delay_calm = rate.compute_delay(arousal=0.0)
        rate2 = AdaptiveCycleRate()
        delay_aroused = rate2.compute_delay(arousal=0.9)
        assert delay_aroused < delay_calm

    def test_rate_clamped_to_limits(self):
        config = AdaptiveRateConfig(min_rate_hz=2.0, max_rate_hz=20.0)
        rate = AdaptiveCycleRate(config=config)
        # Push rate very high
        for _ in range(20):
            rate.compute_delay(input_queue_depth=100, arousal=1.0)
        assert rate.current_rate_hz <= 20.0

    def test_set_idle(self):
        rate = AdaptiveCycleRate()
        delay = rate.set_idle()
        assert rate.current_rate_hz == rate.config.min_rate_hz
        assert delay > 0

    def test_set_active(self):
        rate = AdaptiveCycleRate()
        rate.set_idle()
        rate.set_active()
        assert rate.current_rate_hz == rate.config.base_rate_hz

    def test_rate_timeline(self):
        rate = AdaptiveCycleRate()
        for i in range(10):
            rate.compute_delay()
        timeline = rate.get_rate_timeline()
        assert len(timeline) == 10

    def test_stats(self):
        rate = AdaptiveCycleRate()
        rate.compute_delay()
        stats = rate.get_stats()
        assert stats["total_adjustments"] == 1
        assert stats["current_rate_hz"] > 0


# =========================================================================
# LazyEmbeddingCache
# =========================================================================


class TestLazyEmbeddingCache:
    """Tests for lazy embedding computation."""

    def test_compute_on_first_access(self):
        computed = []
        def compute(text):
            computed.append(text)
            return [0.1, 0.2]

        cache = LazyEmbeddingCache(compute_fn=compute)
        result = cache.get("hello")
        assert result == [0.1, 0.2]
        assert len(computed) == 1

    def test_cache_hit(self):
        computed = []
        def compute(text):
            computed.append(text)
            return [0.1]

        cache = LazyEmbeddingCache(compute_fn=compute)
        cache.get("hello")
        cache.get("hello")
        assert len(computed) == 1  # Only computed once
        assert cache._hits == 1
        assert cache._misses == 1

    def test_cache_miss_different_text(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        e1 = cache.get("hello")
        e2 = cache.get("world")
        assert e1 != e2
        assert cache._misses == 2

    def test_lru_eviction(self):
        config = LazyEmbeddingConfig(max_cache_size=3)
        cache = LazyEmbeddingCache(
            compute_fn=LazyEmbeddingCache.deterministic_test_embedding,
            config=config,
        )
        cache.get("a")
        cache.get("b")
        cache.get("c")
        cache.get("d")  # Should evict "a"
        assert cache._evictions == 1
        assert len(cache._cache) == 3

    def test_ttl_expiration(self):
        config = LazyEmbeddingConfig(ttl_seconds=0.0)  # Immediate expiry
        computed = []
        def compute(text):
            computed.append(text)
            return [0.1]

        cache = LazyEmbeddingCache(compute_fn=compute, config=config)
        cache.get("hello")
        time.sleep(0.01)
        cache.get("hello")  # Should recompute (expired)
        assert len(computed) == 2

    def test_batch_get(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        results = cache.get_batch(["a", "b", "c"])
        assert len(results) == 3
        assert cache._misses == 3

    def test_precompute(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        count = cache.precompute(["a", "b", "c"])
        assert count == 3
        assert cache._misses == 3
        # Now all should be cache hits
        cache.get("a")
        assert cache._hits == 1

    def test_invalidate(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        cache.get("hello")
        assert cache.invalidate("hello") is True
        assert len(cache._cache) == 0

    def test_invalidate_missing(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        assert cache.invalidate("nonexistent") is False

    def test_clear(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        cache.get("a")
        cache.get("b")
        removed = cache.clear()
        assert removed == 2
        assert len(cache._cache) == 0

    def test_disabled_cache(self):
        config = LazyEmbeddingConfig(enabled=False)
        computed = []
        def compute(text):
            computed.append(text)
            return [0.1]

        cache = LazyEmbeddingCache(compute_fn=compute, config=config)
        cache.get("hello")
        cache.get("hello")
        assert len(computed) == 2  # No caching

    def test_hit_rate(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        cache.get("a")  # miss
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.get_hit_rate() == pytest.approx(1 / 3, abs=0.01)

    def test_stats(self):
        cache = LazyEmbeddingCache(compute_fn=LazyEmbeddingCache.deterministic_test_embedding)
        cache.get("hello")
        stats = cache.get_stats()
        assert stats["cache_size"] == 1
        assert stats["misses"] == 1

    def test_deterministic_test_embedding_is_deterministic(self):
        cache = LazyEmbeddingCache(
            compute_fn=LazyEmbeddingCache.deterministic_test_embedding
        )
        e1 = cache.get("test")
        cache.clear()
        e2 = cache.get("test")
        assert e1 == e2

    def test_missing_compute_fn_fails_loud(self):
        """No silent hash fallback: constructing without a real embedder
        must raise, not quietly fabricate embeddings."""
        with pytest.raises(ValueError):
            LazyEmbeddingCache()


# =========================================================================
# AsyncSubsystemProcessor
# =========================================================================


class TestAsyncSubsystemProcessor:
    """Tests for async subsystem processing."""

    @pytest.mark.asyncio
    async def test_run_single_subsystem(self):
        processor = AsyncSubsystemProcessor()

        async def my_task(**kwargs):
            return "done"

        processor.register("task1", my_task)
        results = await processor.run_all()
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].result == "done"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        processor = AsyncSubsystemProcessor()
        order = []

        async def task_a(**kwargs):
            order.append("a_start")
            await asyncio.sleep(0.01)
            order.append("a_end")
            return "a"

        async def task_b(**kwargs):
            order.append("b_start")
            await asyncio.sleep(0.01)
            order.append("b_end")
            return "b"

        processor.register("a", task_a)
        processor.register("b", task_b)
        results = await processor.run_all()
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dependency_ordering(self):
        processor = AsyncSubsystemProcessor()
        order = []

        async def task_a(**kwargs):
            order.append("a")
            return "a"

        async def task_b(**kwargs):
            order.append("b")
            return "b"

        processor.register("a", task_a)
        processor.register("b", task_b, depends_on=["a"])
        await processor.run_all()
        assert order.index("a") < order.index("b")

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        processor = AsyncSubsystemProcessor()

        async def slow_task(**kwargs):
            await asyncio.sleep(10)

        processor.register("slow", slow_task, timeout_ms=50)
        results = await processor.run_all()
        assert len(results) == 1
        assert results[0].success is False
        assert "Timeout" in results[0].error

    @pytest.mark.asyncio
    async def test_error_handling(self):
        processor = AsyncSubsystemProcessor()

        async def failing_task(**kwargs):
            raise ValueError("oops")

        processor.register("fail", failing_task)
        results = await processor.run_all()
        assert results[0].success is False
        assert "oops" in results[0].error

    @pytest.mark.asyncio
    async def test_run_one(self):
        processor = AsyncSubsystemProcessor()

        async def my_task(**kwargs):
            return 42

        processor.register("task", my_task)
        result = await processor.run_one("task")
        assert result.success is True
        assert result.result == 42

    @pytest.mark.asyncio
    async def test_run_one_missing(self):
        processor = AsyncSubsystemProcessor()
        result = await processor.run_one("nonexistent")
        assert result is None

    def test_unregister(self):
        processor = AsyncSubsystemProcessor()

        async def my_task(**kwargs):
            pass

        processor.register("task", my_task)
        assert processor.unregister("task") is True
        assert processor.unregister("task") is False

    def test_execution_order(self):
        processor = AsyncSubsystemProcessor()

        async def t(**kwargs):
            pass

        processor.register("a", t)
        processor.register("b", t, depends_on=["a"])
        processor.register("c", t)
        order = processor.get_execution_order()
        # a and c should be at level 0, b at level 1
        assert len(order) == 2
        assert "b" in order[1]

    @pytest.mark.asyncio
    async def test_execution_history(self):
        processor = AsyncSubsystemProcessor()

        async def t(**kwargs):
            pass

        processor.register("task", t)
        await processor.run_all()
        assert len(processor._execution_history) == 1
        assert processor._execution_history[0]["success_count"] == 1

    @pytest.mark.asyncio
    async def test_duration_tracking(self):
        processor = AsyncSubsystemProcessor()

        async def slow(**kwargs):
            await asyncio.sleep(0.01)

        processor.register("slow", slow)
        results = await processor.run_all()
        assert results[0].duration_ms > 0

    @pytest.mark.asyncio
    async def test_empty_processor(self):
        processor = AsyncSubsystemProcessor()
        results = await processor.run_all()
        assert results == []

    def test_stats(self):
        processor = AsyncSubsystemProcessor()

        async def t(**kwargs):
            pass

        processor.register("task", t)
        stats = processor.get_stats()
        assert stats["registered_subsystems"] == 1
