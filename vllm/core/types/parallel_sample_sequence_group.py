



from typing import Optional
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.sequence import SequenceGroup, SequenceGroupBase


class ParallelSampleSequenceGroup(SequenceGroupBase):

    @staticmethod
    def add_request(request_id: str, engine, params, **kwargs):
        original_params = params
        group = ParallelSampleSequenceGroup(request_id)
        seqs = []
        for i in range(original_params.n):
            request_id_i = f"{request_id}_parallel_sample_{i}"
            group.seq_id_to_index[request_id_i] = i
            params = original_params.clone()
            params.n = 1
            if params.seed is not None:
                params.seed += i
            seq_group = engine._add_processed_request(
                request_id_i,
                params=params,
                **kwargs,
            )  # type: ignore
            assert seq_group is not None
            engine.seq_id_to_seq_group[request_id_i] = group
            group.to_be_finished[request_id_i] = seq_group
            seqs.append(seq_group.seqs[0])

        # for parallel sampling, the `assembled_seq_group` is always
        # available, since we have all the sequences ready, and they
        # will not change.
        group.assembled_seq_group = SequenceGroup(
            request_id=request_id,
            seqs=seqs,
            arrival_time=seq_group.arrival_time,
            sampling_params=original_params,
            lora_request=seq_group.lora_request,
            pooling_params=seq_group.pooling_params,
            pooled_data=seq_group.pooled_data,
            encoder_seq=seq_group.encoder_seq,
            trace_headers=seq_group.trace_headers,
            priority=seq_group.priority,
        )

        group.streaming = params.output_kind == RequestOutputKind.DELTA
        group.output_produced = False

    def maybe_assemble_group(
            self, seq_group: SequenceGroup) -> Optional[SequenceGroup]:

        # in the streaming mode, we will return the assembled sequence
        # for the first remaining sequence, and then return None for the
        # rest of sequences
        if self.streaming:
            first_remaining_id = next(iter(self.to_be_finished))
            if seq_group.request_id == first_remaining_id:
                return self.assembled_seq_group
            return None

        # in the non-streaming mode, we will return the assembled sequence
        # when the last sequences finishes, and then return None for the
        # rest of the time
        if (len(self.to_be_finished) == 1
                and seq_group.request_id in self.to_be_finished
                and seq_group.is_finished()):
            assert self.assembled_seq_group is not None
            params = self.assembled_seq_group.sampling_params
            assert isinstance(params, SamplingParams)
            if not self.output_produced:
                self.output_produced = True
                if params._real_n is not None:
                    # Get the top-n sequences.
                    n = params._real_n or params.n
                    seqs = self.assembled_seq_group.seqs
                    sorting_key = lambda seq: seq.get_cumulative_logprob()
                    sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
                    top_n_seqs = sorted_seqs[:n]
                    self.assembled_seq_group.seqs = top_n_seqs
                return self.assembled_seq_group
            if self.output_produced:
                return None
        return None
