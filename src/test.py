import torch
import torch.nn.functional as F
from tqdm import tqdm

def test_epoch_ver(model, pair_data_loader, device):
    model.eval()
    scores = []
    batch_bar = tqdm(total=len(pair_data_loader), dynamic_ncols=True, position=0, leave=False, desc='Test Veri.')
    
    for i, (images1, images2) in enumerate(pair_data_loader):
        images = torch.cat([images1, images2], dim=0).to(device)
        
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs['feats'], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.extend(similarity.cpu().numpy().tolist())
        batch_bar.update()

    return scores

def generate_submission(scores, output_file):
    with open(output_file, "w+") as f:
        f.write("ID,Label\n")
        for i, score in enumerate(scores):
            f.write(f"{i},{score}\n")
    print(f"Submission file created: {output_file}")