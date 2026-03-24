#!/usr/bin/env python3
"""
Lightweight synthetic tests for ImageNet FracBNN models.
"""

import os
import sys

import torch
import torchvision

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model.fracbnn_imagenet as m
from utils.quantization import KnowledgeDistillationLoss


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def make_normalized_input(batch_size, device):
    raw = torch.rand(batch_size, 3, 224, 224, device=device)
    return (raw - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def test_forward_paths():
    print("=" * 80)
    print("Testing ImageNet Model Forward Paths")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_passed = True

    for adaptive_pg in (False, True):
        name = 'Adaptive PG ReActNet' if adaptive_pg else 'Baseline ReActNet'
        print(f"\n{name}")

        try:
            model = m.ReActNet(
                batch_size=2,
                num_gpus=1,
                adaptive_pg=adaptive_pg,
                target_sparsity=0.15,
            ).to(device)
            model.eval()

            x = make_normalized_input(2, device)
            with torch.no_grad():
                out = model(x)

            assert out.shape == (2, 1000), f"Expected (2, 1000), got {out.shape}"

            if adaptive_pg:
                stats = model.get_gate_statistics()
                assert stats, "Expected adaptive PG layers to report gate stats"
                print(f"  Gate layers: {len(stats)}")
                print(f"  Sparsity loss: {model.get_sparsity_loss().item():.6f}")

            print(f"  Output shape: {out.shape}")
            print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
            print("  PASSED")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            all_passed = False

    return all_passed


def test_single_kd_step():
    print("\n" + "=" * 80)
    print("Testing Single ImageNet KD Training Step")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        student = m.ReActNet(
            batch_size=1,
            num_gpus=1,
            adaptive_pg=True,
            target_sparsity=0.15,
        ).to(device)
        teacher = torchvision.models.resnet18(weights=None).to(device)
        teacher.eval()

        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
        kd_loss = KnowledgeDistillationLoss(
            temperature=4.0,
            alpha=0.7,
            label_smoothing=0.1,
        ).to(device)

        x = make_normalized_input(1, device)
        labels = torch.randint(0, 1000, (1,), device=device)

        student.train()
        optimizer.zero_grad()
        student_logits = student(x)
        with torch.no_grad():
            teacher_logits = teacher(x)

        loss = kd_loss(student_logits, teacher_logits, labels)
        loss = loss + 0.01 * student.get_sparsity_loss() + 0.001 * student.get_entropy_loss()
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss), "Loss became NaN or Inf"
        print(f"Loss: {loss.item():.4f}")
        print("PASSED")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def main():
    results = {
        'Forward Paths': test_forward_paths(),
        'Single KD Step': test_single_kd_step(),
    }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name:.<40} {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
