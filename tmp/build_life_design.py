from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from pypdf import PdfReader

out = Path('output/pdf/life_preserved_design.pdf')
out.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyCompact', fontName='Helvetica', fontSize=9.5,
                         leading=13, spaceAfter=7, textColor=colors.HexColor('#253044')))
styles.add(ParagraphStyle(name='SectionCompact', fontName='Helvetica-Bold', fontSize=10,
                         leading=13, spaceBefore=5, spaceAfter=3))
content = [Paragraph('LIFE Preserved prediction integration', styles['Title']),
           Paragraph('Design principles applied to single assessments and resumable batches', styles['BodyCompact'])]
sections = [
('1 Define what is being built',
 'For researchers comparing LIFE-Preserved with existing predictions, the integration maps the accepted cohort fields to the public calculator and retains every assessment output. The flow is cohort row to validated profile to browser calculation to local result. Source proxies remain as accepted.'),
('2 Design the user experience',
 'Python functions support one profile or a DataFrame batch; the command line supports limits, units, defaults, treatments and resume. Each result includes inputs and imputed fields. Invalid rows are recorded as errors while the batch continues. Successful unchanged rows are skipped on restart.'),
('3 Understand the technical needs',
 'Small functions separate mapping, encoding, reference statistics, Selenium setup, form filling, extraction and checkpointing. Drivers and prediction callables are injected, separating creation from use and allowing offline substitutes without a class hierarchy. Selenium visits all three tabs to collect nine labelled numeric values, three PNG plots and rendered HTML. Imports perform no data loading or browser work.'),
('4 Implement testing and security measures',
 'Regression tests cover source versus website sex coding, missingness, haemoglobin units, bounds, parsing, retries, resume and row isolation. Only required predictor values reach the website; row identifiers stay local. Fresh page loads and a calculation completion event prevent stale results, including identical consecutive profiles. Browser sessions close in a finally block.'),
('5 Plan the work',
 'Milestones are live form discovery, pure input preparation, single assessment extraction, batch recovery and verification. Completion requires all nine values and three plots or an explicit row error. Network and DOM changes are the main integration risks; bounded retries and checkpoints preserve progress. OOF comparison remains a separate later step. No database migration is needed.'),
('6 Identify ripple effects',
 'The cohort script now has a main entry point, stable file paths and documented CLI options. A separate requirements file adds Selenium. JSON checkpoints and a flattened CSV make results available for analysis; JSON records the treatment scenario, input defaults and units. Reference means and categorical modes can be fitted on an explicit training cohort.'),
('7 Understand the broader context',
 'Entirely absent fields use configurable reference assumptions, not measured population means: heart rate 70 bpm, NYHA I/II, no recent HF admission and non-smoking. Other fallbacks are age 70, female, BMI 25, NT-proBNP 300 pg/mL, eGFR 75 and haemoglobin 14 g/dL; binary histories default to no. Values outside website ranges fail rather than being clipped. The website labels results preliminary and not intended for clinical use. A future official API could replace the injected browser transport.'),
]
for heading, body in sections:
    content += [Paragraph(heading, styles['SectionCompact']), Paragraph(body, styles['BodyCompact'])]
content.append(Spacer(1, 6))
content.append(Paragraph('Framework: ArjanCodes Software Design Guide, supplied by the user. Implementation: life_preserved.py and run_life_comparison.py. 25 October 2026.',
                         ParagraphStyle(name='Source',fontName='Helvetica',fontSize=8,leading=10,textColor=colors.HexColor('#586174'))))
SimpleDocTemplate(str(out), pagesize=(595.28,841.89), leftMargin=42,rightMargin=42,
                  topMargin=30,bottomMargin=28,title='LIFE Preserved prediction integration').build(content)
assert len(PdfReader(out).pages) == 1, 'Design document must be one page'
print(out.resolve())
