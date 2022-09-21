from src.decoder import local_decoder

decoder_dict = {
    'local_decoder': local_decoder.LocalDecoder,
    'local_decoder_Larger': local_decoder.LocalDecoderLarger,
    # 'simple_local_crop': decoder.PatchLocalDecoder,
    'local_with_probe_decoder': local_decoder.WithProbeDecoder,
    'local_with_probe_decoder': local_decoder.WithProbeDecoder,
    
}