"""Small synthetic model; replace calibration with representative real inputs."""
import argparse
import torch
import torch_coral


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/basic')
    args = parser.parse_args()
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)).eval()
    data = [torch.rand(1, 4) for _ in range(64)]
    with torch_coral.compile(model, data[0], data, artifact_dir=args.output) as coral:
        with torch.no_grad():
            result = coral(data[0])
        print(result)
        print(coral.manifest['compiler_report'])


if __name__ == '__main__': main()
