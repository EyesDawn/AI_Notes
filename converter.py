from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import time


start_time = time.time()

source = "D:\\projects\\AI\\Notes\\diffusion\\The Rise of Diffusion Models in Time-Series Forecasting.pdf"  # PDF path or URL
pipeline_options = PdfPipelineOptions()
pipeline_options.do_formula_enrichment = True
pipeline_options.do_ocr = False

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
})
result = converter.convert(source)
output_path = "D:\\projects\\AI\\Notes\\diffusion\\The Rise of Diffusion Models in Time-Series Forecasting.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(result.document.export_to_markdown())
print(f"Markdown file saved to {output_path}")

end_time = time.time()
print(f"Conversion took {end_time - start_time:.2f} seconds.")