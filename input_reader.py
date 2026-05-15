from pathlib import Path
import pandas as pd
from config import SUPPORTED_IMAGE_EXTS, SUPPORTED_PDF_EXTS, SUPPORTED_TEXT_EXTS


def read_input_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Input file not found: {file_path}')
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
    if suffix in SUPPORTED_TEXT_EXTS:
        return read_text_file(path)
    if suffix in SUPPORTED_PDF_EXTS:
        return read_pdf_text(path)
    if suffix in SUPPORTED_IMAGE_EXTS:
        return read_image_ocr(path)
    raise ValueError(f'Unsupported file format: {suffix}')


def read_text_file(path: Path) -> pd.DataFrame:
    lines = [line.strip() for line in path.read_text(encoding='utf-8', errors='ignore').splitlines() if line.strip()]
    return pd.DataFrame({'text': lines, 'source_file': path.name})


def read_pdf_text(path: Path) -> pd.DataFrame:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError('Install PDF dependencies first: pip install pdfplumber') from exc
    rows = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ''
            for line in text.splitlines():
                line = line.strip()
                if line:
                    rows.append({'text': line, 'page': page_number, 'source_file': path.name})
    return pd.DataFrame(rows)


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
