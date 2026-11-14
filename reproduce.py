"""Build and verify the GitHub-ready maize analysis archive."""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG6 = 'Fig6_ML_prioritization_mechanism_inference'
SUPP = ['SuppFig_FT_01_OOF_validation', 'SuppFig_FT_02_candidate_robustness',
        'SuppFig_FT_03_mechanism_coverage', 'SuppFig_FT_04_network_sensitivity']
EXTRA = ['Fig_top3_nested_L1_L2_mechanism_priority_source_data.csv',
         'Fig_top3_candidate_mechanism_evidence_paths_source_edges.tsv']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def build(source, dest, readme):
    dest.mkdir()
    copy = lambda a, b: shutil.copy2(source / a, dest / b)
    analysis = dest / 'analysis'
    analysis.mkdir()
    old = source / 'ML_prediction_script'
    shutil.copy2(old / 'run_all.py', analysis)
    shutil.copy2(old / 'requirements.txt', dest)
    for module in ['function_prediction', 'mechanism_prediction']:
        shutil.copytree(old / module, analysis / module,
                        ignore=shutil.ignore_patterns('__pycache__', 'provenance', 'upstream', '*.pyc'))
    config = json.loads((old / 'config.example.json').read_text(encoding='utf-8'))
    config.update(data_root='../inputs', results_root='../outputs/analysis')
    config['mechanism_prediction']['input_dir'] = '../outputs/Mechanism_input'
    write_json(analysis / 'config.json', config)
    (dest / 'inputs/network_parts').mkdir(parents=True)
    for name in ['FT_training_feature_matrix.csv', 'candidate9_feature_matrix.csv',
                 'maize_FT_training_labels_v5.tsv', 'List_of_9_candidate_genes_to_be_predicted.csv']:
        copy(name, 'inputs/' + name)
    archive = {}
    with tempfile.TemporaryDirectory(dir=dest) as temp:
        gz = Path(temp) / 'input.gz'
        for p in sorted((source / 'Mechanism_input').glob('*.tsv')):
            print('Packing', p.name, flush=True)
            h = hashlib.sha256()
            with p.open('rb') as src, gz.open('wb') as raw:
                with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=4, mtime=0) as out:
                    while chunk := src.read(4 * 1024**2):
                        h.update(chunk)
                        out.write(chunk)
            pieces = []
            with gz.open('rb') as f:
                while chunk := f.read(20 * 1024**2):
                    part = 'inputs/network_parts/' + p.name + f'.gz.part{len(pieces):03}'
                    (dest / part).write_bytes(chunk)
                    pieces.append(part)
            archive[p.name] = {'parts': pieces, 'sha256': h.hexdigest(), 'bytes': p.stat().st_size}
    assert len(archive) == 11
    write_json(dest / 'inputs/mechanism_archive.json', archive)
    for name, script in [('Fig6','make_fig6.py'), ('Supplementary_Figures','make_supplementary_figures.py')]:
        shutil.copytree(source / name / 'source_data', dest / 'figure_data' / name / 'source_data')
        (dest / 'plotting').mkdir(exist_ok=True)
        copy(f'{name}/{script}', f'plotting/{script}')
    (dest / 'plotting/qa_tools').mkdir()
    copy('Fig6/qa_tools/audit_panel_alignment.py','plotting/qa_tools/audit_panel_alignment.py')
    for entry in EXTRA:
        copy(entry, 'figure_data/' + entry)
    copy('Supplementary_Figures/Supplementary_figure_legends.md','figure_data/Supplementary_figure_legends.md')
    shutil.copy2(__file__, dest / 'reproduce.py')
    shutil.copy2(readme, dest / 'README.md')
    (dest / '.gitignore').write_text('/outputs/\n/.venv/\n__pycache__/\n*.pyc\n', encoding='utf-8')
    files = sorted(p for p in dest.rglob('*') if p.is_file())
    write_json(dest / 'SHA256.json', {p.relative_to(dest).as_posix():digest(p) for p in files})
    assert max(p.stat().st_size for p in files) < 25_000_000


def unpack():
    manifest = json.loads((HERE / 'inputs/mechanism_archive.json').read_text())
    target = HERE / 'outputs/Mechanism_input'
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as temp:
        tmp = Path(temp)
        for name, row in manifest.items():
            p = target / name
            if p.is_file() and digest(p) == row['sha256']:
                continue
            gz, plain = tmp / 'archive.gz', tmp / 'unpacked.tsv'
            with gz.open('wb') as dst:
                for part in row['parts']:
                    with (HERE / part).open('rb') as src:
                        shutil.copyfileobj(src, dst)
            with gzip.open(gz, 'rb') as src, plain.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            if digest(plain) != row['sha256']:
                raise ValueError(f'SHA256 mismatch: {name}')
            plain.replace(p)


def call(script, *args):
    subprocess.run([sys.executable, '-B', str(HERE / script), *map(str,args)], check=True)


def redraw():
    qa = HERE / 'plotting/qa_tools'
    data = HERE / 'figure_data'
    # Keep the original shared crop; discard the uppercase rendering.
    with tempfile.TemporaryDirectory() as temp:
        call('plotting/make_fig6.py', '--root', data, '--outdir', HERE/'outputs/figures/Fig6',
             '--capital-outdir', temp, '--qa-tools', qa)
        call('plotting/make_supplementary_figures.py', '--root', data/'Supplementary_Figures',
             '--out', HERE/'outputs/figures/Supplementary_Figures', '--capital-out', temp,
             '--qa-tools', qa)


def check():
    import pandas as pd
    from pandas.testing import assert_frame_equal
    root = HERE/'outputs/analysis'
    frozen = HERE/'figure_data/Supplementary_Figures/source_data'
    tests = {}
    for name in ['FT_model_comparison.csv','FT_outer_fold_metrics.csv','FT_oof_predictions.csv',
                 'FT_calibration_curve.csv','candidate9_predictions.csv',
                 'candidate9_all_model_summary.csv','candidate9_feature_block_contributions.csv']:
        a = pd.read_csv(frozen/name)
        b = pd.read_csv(root/'function_prediction'/name)
        assert_frame_equal(a,b,check_dtype=False,rtol=1e-9,atol=1e-10)
        tests[name] = len(a)
    name = 'candidate9_mechanism_results_v2.json'
    a = json.loads((frozen/'mechanism_plot_data.json').read_text(encoding='utf-8'))
    b = json.loads((root/'mechanism_prediction'/name).read_text(encoding='utf-8'))
    for key in ['candidate_summary','l1_scores_long','coverage_qc']:
        x,y = pd.DataFrame(a[key]),pd.DataFrame(b[key])
        cols = [c for c in x.columns if c in y.columns]
        assert cols
        assert_frame_equal(x[cols],y[cols],check_dtype=False,rtol=1e-8,atol=1e-8)
        tests[key] = {'rows':len(x),'compared_columns':cols}
    write_json(HERE/'outputs/analysis_comparison.json',tests)
    print('Archived analysis values matched:', tests)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['build','verify','unpack','analyze','check','redraw','all'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--dest',type=Path)
    parser.add_argument('--readme',type=Path)
    args = parser.parse_args()
    if args.mode == 'build':
        build(args.source,args.dest,args.readme)
    elif args.mode == 'verify':
        rows = json.loads((HERE/'SHA256.json').read_text())
        for name, expected in rows.items():
            if digest(HERE/name) != expected:
                raise ValueError(f'SHA256 mismatch: {name}')
        print(f'Verified {len(rows)} files')
    else:
        if args.mode in ['unpack','analyze','all']:
            unpack()
        if args.mode in ['analyze','all']:
            call('analysis/run_all.py','--config',HERE/'analysis/config.json')
        if args.mode in ['check','all']:
            check()
        if args.mode in ['redraw','all']:
            redraw()


if __name__ == '__main__':
    main()
