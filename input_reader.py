from pathlib import Path
import pandas as pd
from config import SUPPORTED_IMAGE_EXTS


def read_input_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        frames = []
        for sheet_name, df in sheets.items():
            df = df.copy()
            df['source_sheet'] = sheet_name
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if suffix == '.csv':
        return pd.read_csv(path, dtype=str)
    if suffix in SUPPORTED_IMAGE_EXTS:
        return read_image_ocr(path)
    raise ValueError(f'Unsupported file format: {suffix}')


def read_image_ocr(path: Path) -> pd.DataFrame:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise ImportError('Install OCR dependencies first: pip install paddleocr paddlepaddle') from exc
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    result = ocr.ocr(str(path), cls=True)
    rows = []
    for page in result:
        for item in page:
            box, text_score = item
            text, score = text_score
            rows.append({
                'text': text,
                'score': score,
                'x': min(p[0] for p in box),
                'y': min(p[1] for p in box),
                'source_file': path.name,
            })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    return ocr_tokens_to_table(raw)


def ocr_tokens_to_table(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.sort_values(['y', 'x']).reset_index(drop=True)
    line_threshold = 18
    lines, current, current_y = [], [], None
    for _, row in raw.iterrows():
        if current_y is None or abs(row['y'] - current_y) <= line_threshold:
            current.append(row)
            current_y = row['y'] if current_y is None else current_y
        else:
            lines.append(current)
            current, current_y = [row], row['y']
    if current:
        lines.append(current)
    parsed = [[r['text'] for r in sorted(line, key=lambda v: v['x'])] for line in lines]
    max_cols = max(len(r) for r in parsed) if parsed else 0
    return pd.DataFrame([r + [''] * (max_cols - len(r)) for r in parsed])
