# Generate a PDF payment receipt from a list of line items.
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

TAX_RATE = Decimal("0.10")

# Line items: (date, name, subscription, amount)
LINE_ITEMS = [
    ("2024-06-15", "John Doe", "Premium Plan", Decimal("49.99")),
    ("2024-07-15", "Jane Smith", "Basic Plan", Decimal("19.99")),
]


def money(amount: Decimal) -> str:
    return f"${amount.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):,}"


def build_rows(items: list[tuple[str, str, str, Decimal]]) -> list[list[str]]:
    # Totals are derived from the line items so they can never disagree with them.
    subtotal = sum((amount for *_, amount in items), Decimal("0"))
    tax = (subtotal * TAX_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    rows = [["Date", "Name", "Subscription", "Amount"]]
    rows += [[date, name, plan, money(amount)] for date, name, plan, amount in items]
    rows += [
        ["Subtotal", "", "", money(subtotal)],
        [f"Tax ({TAX_RATE:.0%})", "", "", money(tax)],
        ["Total", "", "", money(subtotal + tax)],
    ]
    return rows


def build_receipt(pdf_file: Path) -> None:
    document = SimpleDocTemplate(str(pdf_file), pagesize=A4)
    styles = getSampleStyleSheet()

    table = Table(build_rows(LINE_ITEMS))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    document.build([
        Paragraph("Payment Receipt", styles["Title"]),
        Spacer(1, 12),
        table,
        Spacer(1, 12),
        Paragraph("Thank you for your payment!", styles["Normal"]),
    ])


def main() -> None:
    pdf_file = Path(__file__).parent / "outputs" / "payment_receipt.pdf"
    pdf_file.parent.mkdir(parents=True, exist_ok=True)
    build_receipt(pdf_file)
    print(f"Payment receipt generated: {pdf_file}")


if __name__ == "__main__":
    main()
