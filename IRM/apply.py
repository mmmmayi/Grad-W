from torch.utils.data import DataLoader
from Models.dataset import IRMDataset
from Trainer.applier import IRMApplier


PROJECT_DIR = "exp/dysnr_enhanceDisto1e8norm_BLSTM_frame_0.001_adadelta_4s_wada0.1_th0.1/wav_100"
MODEL_PATH = "exp/dysnr_enhanceDisto1e8norm_BLSTM_frame_0.001_adadelta_4s_wada0.1_th0.1/models/model_100.pt"
mode = 'quality'

if __name__ == "__main__":
    ## Dataloader
    applier_irm_dataset = IRMDataset(
        path="/data_a11/mayi/project/CAM/lst/utt_dns",
        spk="/data_a11/mayi/project/CAM/lst/val_spk.lst",
        batch_size=1,dur=0,
        sampling_rate=16000, mode=mode, max_size=10000, data=mode)

    applier_loader = DataLoader(
        dataset=applier_irm_dataset,
        batch_size=1,
        shuffle=None,
        num_workers=2)

    irm_applier = IRMApplier(
        project_dir=PROJECT_DIR,
        test_set_name="BATCH_D",
        sampling_rate=16000,
        model_path=MODEL_PATH,
        applier_dl=applier_loader,
        mode = mode)
    irm_applier.restore()
    irm_applier.quality()
