import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer


class TestDeepseekV4DSparkTpEp(unittest.TestCase):
    def test_uniform_verify_metadata_matches_tp_padded_cache(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        class MetadataChecked(Exception):
            pass

        def check_core(**kwargs):
            self.assertEqual(kwargs["out_loc"].numel(), 8)
            self.assertEqual(kwargs["seq_lens_casual"].tolist(), [11, 12, 13, 14, 15, 16, 1, 1])
            self.assertEqual(kwargs["req_pool_indices_repeated"].tolist(), [2, 2, 2, 2, 2, 2, 0, 0])
            raise MetadataChecked

        backend = SimpleNamespace(
            speculative_num_draft_tokens=6,
            req_to_token=None,
            MAX_SEQ_LEN_FOR_CAPTURE=128,
            cuda_int32_kwargs={"device": "cpu", "dtype": torch.int32},
            make_core_attn_metadata=check_core,
        )
        backend.expand_extend_with_same_length = (
            lambda **kwargs: DeepseekV4AttnBackend.expand_extend_with_same_length(
                backend, **kwargs
            )
        )
        raw = SimpleNamespace(
            req_pool_indices=torch.tensor([2]),
            seq_lens=torch.tensor([10]),
            out_cache_loc=torch.arange(8),
            extend_seq_lens=torch.tensor([6]),
            verify_lens=None,
        )
        with self.assertRaises(MetadataChecked):
            DeepseekV4AttnBackend.make_forward_metadata_from_raw_verify(backend, raw)

    def test_hip_multimem_uses_collective_fallback_without_building(self):
        from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
            MultimemAllGatherer,
        )

        x = torch.ones(2, 4, dtype=torch.bfloat16)
        with (
            patch.object(torch.version, "hip", "test-hip"),
            patch.object(MultimemAllGatherer, "_build") as build,
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_gather",
                return_value=x,
            ) as gather,
        ):
            runner = MultimemAllGatherer(max_tokens=128, enabled=True)
            self.assertIs(runner(x), x)
            build.assert_not_called()
            gather.assert_called_once_with(x, dim=-1)

    def test_tp_a2a_scatter_accepts_missing_dspark_token_ids(self):
        mlp_inputs = []

        def mlp(hidden_states, _forward_batch, **kwargs):
            mlp_inputs.append(kwargs)
            return hidden_states

        layer = SimpleNamespace(dsa_enable_prefill_cp=False, mlp=mlp)
        parallel = SimpleNamespace(
            attn_dp_size=1,
            attn_tp_size=2,
            attn_tp_rank=0,
            tp_size=2,
        )
        a2a_backend = SimpleNamespace(is_none=lambda: False)
        forward_context = SimpleNamespace(scoped=lambda **_kwargs: nullcontext())

        def all_gather(outputs, local_hidden_states):
            for output in outputs:
                output.copy_(local_hidden_states)

        hidden_states = torch.arange(8, dtype=torch.float32).view(4, 2)
        with (
            patch("sglang.srt.models.deepseek_v4.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.models.deepseek_v4.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.get_forward",
                return_value=forward_context,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.attn_tp_all_gather",
                side_effect=all_gather,
            ),
        ):
            output = DeepseekV4DecoderLayer._run_moe_ffn_dp_sync(
                layer,
                hidden_states,
                SimpleNamespace(dp_padding_mode=None),
                input_ids=None,
                input_ids_global=None,
            )

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIsNone(mlp_inputs[0]["input_ids"])
        self.assertIsNone(mlp_inputs[0]["input_ids_global"])


if __name__ == "__main__":
    unittest.main()
