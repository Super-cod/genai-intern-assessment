"""Generate 12 sample invoices with different layouts + ground truth, and invalid-input edge cases.

All companies are fictional. Run:  python samples/generate_samples.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
EDGE = HERE / "edge_cases"
W, H = pymupdf.paper_size("a4")
FONT = "helv"
BOLD = "hebo"


class Page:
    def __init__(self, doc):
        self.p = doc.new_page(width=W, height=H)

    def t(self, x, y, text, size=10, bold=False, color=(0, 0, 0)):
        for i, line in enumerate(text.split("\n")):
            self.p.insert_text((x, y + i * (size + 3)), line, fontsize=size, fontname=BOLD if bold else FONT,
                               color=color)

    def rt(self, x_right, y, text, size=10, bold=False):  # right-aligned
        w = pymupdf.get_text_length(text, fontname=BOLD if bold else FONT, fontsize=size)
        self.t(x_right - w, y, text, size, bold)

    def line(self, x0, y0, x1, y1, width=0.6):
        self.p.draw_line((x0, y0), (x1, y1), width=width, color=(0.5, 0.5, 0.5))

    def table(self, x, y, headers, rows, col_x, right_cols=()):
        for h, cx in zip(headers, col_x):
            self.t(x + cx, y, h, 9.5, bold=True)
        self.line(x, y + 4, W - 50, y + 4)
        y += 18
        for row in rows:
            for i, (cell, cx) in enumerate(zip(row, col_x)):
                if i in right_cols:
                    nxt = col_x[i + 1] if i + 1 < len(col_x) else (W - 50 - x)
                    self.rt(x + nxt - 8, y, cell, 9.5)
                else:
                    self.t(x + cx, y, cell, 9.5)
            y += 16
        self.line(x, y - 8, W - 50, y - 8)
        return y


def items_rows(items, fmt):
    return [[d, str(q), fmt(p), fmt(q * p)] for d, q, p in items]


def us(v):
    return f"{v:,.2f}"


def eu(v):
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def inr(v):
    s = f"{v:.2f}"
    whole, dec = s.split(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    return f"{whole}.{dec}"


def gt_items(items):
    return [{"description": d, "quantity": q, "unit_price": round(p, 2), "amount": round(q * p, 2)} for d, q, p in items]


GROUND_TRUTH: dict[str, dict] = {}


def save(doc, name, truth):
    doc.save(HERE / name)
    GROUND_TRUTH[name] = truth


# ----------------------------------------------------------------------------- 1 classic USD
def inv01():
    items = [("Ergonomic office chair", 5, 180.0), ("Standing desk frame", 2, 150.0), ("Cable management kit", 10, 5.0)]
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Brightline Supplies LLC", 16, True)
    pg.t(50, 80, "450 Market Street\nDenver, CO 80202\nEIN: 84-1234567")
    pg.rt(W - 50, 60, "INVOICE", 22, True)
    pg.rt(W - 50, 85, "Invoice No: INV-2024-0012"); pg.rt(W - 50, 99, "Invoice Date: March 5, 2024")
    pg.rt(W - 50, 113, "Due Date: April 4, 2024"); pg.rt(W - 50, 127, "PO Number: PO-7781")
    pg.t(50, 160, "Bill To:", 10, True); pg.t(50, 175, "Northfield Clinic\n12 Elm Road\nBoulder, CO 80301")
    y = pg.table(50, 240, ["Description", "Qty", "Unit Price", "Amount"], items_rows(items, us), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 10, "Subtotal: USD 1,250.00"); pg.rt(W - 50, y + 26, "Sales Tax (8%): USD 100.00")
    pg.rt(W - 50, y + 46, "Total Due: USD 1,350.00", 12, True)
    pg.t(50, y + 90, "Payment Terms: Net 30. Please include the invoice number with your payment.")
    save(doc, "inv01_classic_usd.pdf", dict(
        invoice_number="INV-2024-0012", invoice_date="2024-03-05", due_date="2024-04-04",
        vendor_name="Brightline Supplies LLC", vendor_address="450 Market Street, Denver, CO 80202",
        vendor_tax_id="84-1234567", customer_name="Northfield Clinic", customer_address="12 Elm Road, Boulder, CO 80301",
        purchase_order="PO-7781", currency="USD", subtotal=1250.00, tax_amount=100.00, total_amount=1350.00,
        payment_terms="Net 30", line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 2 EU two-column
def inv02():
    items = [("CNC spindle maintenance", 1, 1450.0), ("Replacement bearings set", 4, 250.0)]
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 55, "Kessler Maschinenbau GmbH", 15, True)
    pg.t(50, 72, "Industriestrasse 14, 70565 Stuttgart, Germany", 9, color=(0.3, 0.3, 0.3))
    pg.t(50, 120, "FROM", 9, True); pg.t(50, 134, "Kessler Maschinenbau GmbH\nIndustriestrasse 14\n70565 Stuttgart\nVAT ID: DE123456789")
    pg.t(320, 120, "INVOICE TO", 9, True); pg.t(320, 134, "Van Dijk Logistics B.V.\nHavenweg 3\n3011 Rotterdam, Netherlands")
    pg.t(50, 215, "Invoice no.: 2024/118"); pg.t(220, 215, "Date: 12.02.2024"); pg.t(380, 215, "Due: 13.03.2024")
    y = pg.table(50, 250, ["Item", "Qty", "Price (EUR)", "Total (EUR)"], items_rows(items, eu), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 10, "Net amount: 2.450,00 EUR"); pg.rt(W - 50, y + 26, "VAT 19%: 465,50 EUR")
    pg.rt(W - 50, y + 46, "Gross total: 2.915,50 EUR", 12, True)
    pg.t(50, y + 90, "Terms: payable within 30 days without deduction.")
    save(doc, "inv02_eu_two_column.pdf", dict(
        invoice_number="2024/118", invoice_date="2024-02-12", due_date="2024-03-13",
        vendor_name="Kessler Maschinenbau GmbH", vendor_address="Industriestrasse 14, 70565 Stuttgart",
        vendor_tax_id="DE123456789", customer_name="Van Dijk Logistics B.V.",
        customer_address="Havenweg 3, 3011 Rotterdam, Netherlands", purchase_order=None, currency="EUR",
        subtotal=2450.00, tax_amount=465.50, total_amount=2915.50,
        payment_terms="payable within 30 days without deduction", line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 3 missing fields
def inv03():
    items = [("Website hosting (annual)", 1, 240.0), ("Domain renewal", 2, 15.0)]
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Pixelforge Studio", 16, True); pg.t(50, 80, "88 Harbor Lane, Portland, OR 97201")
    pg.t(50, 120, "INVOICE #PF-0931", 14, True); pg.t(50, 138, "Issued: 2024-06-18")
    pg.t(50, 170, "Client: Greenleaf Bakery")
    y = pg.table(50, 210, ["Service", "Qty", "Rate", "Line total"], items_rows(items, us), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 20, "TOTAL USD 270.00", 12, True)
    pg.t(50, y + 70, "Thank you for your business!")
    save(doc, "inv03_missing_fields.pdf", dict(
        invoice_number="PF-0931", invoice_date="2024-06-18", due_date=None, vendor_name="Pixelforge Studio",
        vendor_address="88 Harbor Lane, Portland, OR 97201", vendor_tax_id=None, customer_name="Greenleaf Bakery",
        customer_address=None, purchase_order=None, currency="USD", subtotal=None, tax_amount=None,
        total_amount=270.00, payment_terms=None, line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 4 many items GBP
def inv04():
    items = [("A4 copier paper (box)", 12, 24.5), ("Toner cartridge black", 3, 62.0), ("Toner cartridge colour", 3, 74.0),
             ("Whiteboard markers (pack)", 6, 4.25), ("Lever arch files", 20, 2.1), ("Desk organiser", 4, 11.99),
             ("Stapler heavy duty", 2, 18.75), ("Laminating pouches (100)", 5, 9.6)]
    sub = sum(q * p for _, q, p in items); vat = round(sub * 0.2, 2); tot = sub + vat
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Office Hub Ltd", 16, True); pg.t(50, 80, "Unit 7, Riverside Park\nLeeds LS1 4AP\nVAT Reg No: GB 987 6543 21")
    pg.rt(W - 50, 60, "TAX INVOICE", 18, True); pg.rt(W - 50, 82, "Invoice Number: OH-55120")
    pg.rt(W - 50, 96, "Invoice Date: 21 August 2024"); pg.rt(W - 50, 110, "Payment Due: 20 September 2024")
    pg.t(50, 150, "Customer:", 10, True); pg.t(50, 165, "Hartley & Sons Accountants\n22 King Street\nYork YO1 9SP")
    y = pg.table(50, 230, ["Description", "Qty", "Unit £", "Total £"], items_rows(items, us), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 10, f"Subtotal: £{us(sub)}"); pg.rt(W - 50, y + 26, f"VAT @ 20%: £{us(vat)}")
    pg.rt(W - 50, y + 46, f"Amount Due: £{us(tot)}", 12, True)
    save(doc, "inv04_many_items_gbp.pdf", dict(
        invoice_number="OH-55120", invoice_date="2024-08-21", due_date="2024-09-20", vendor_name="Office Hub Ltd",
        vendor_address="Unit 7, Riverside Park, Leeds LS1 4AP", vendor_tax_id="GB 987 6543 21",
        customer_name="Hartley & Sons Accountants", customer_address="22 King Street, York YO1 9SP",
        purchase_order=None, currency="GBP", subtotal=round(sub, 2), tax_amount=vat, total_amount=round(tot, 2),
        payment_terms=None, line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 5 ambiguous date + bare $
def inv05():
    items = [("Event photography (half day)", 1, 600.0), ("Edited photo package", 1, 150.0)]
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Lumen Photo Co.", 16, True); pg.t(50, 80, "301 Queen St W, Suite 4")
    pg.t(50, 120, "Invoice: LP-2207"); pg.t(50, 134, "Date: 03/04/2024")
    pg.t(50, 160, "Billed to: Maple Events Inc.")
    y = pg.table(50, 200, ["Description", "Qty", "Price", "Amount"], items_rows(items, lambda v: "$" + us(v)), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 20, "Total: $750.00", 12, True)
    save(doc, "inv05_ambiguous_date_currency.pdf", dict(
        invoice_number="LP-2207", invoice_date={"status": "ambiguous"}, due_date=None, vendor_name="Lumen Photo Co.",
        vendor_address="301 Queen St W, Suite 4", vendor_tax_id=None, customer_name="Maple Events Inc.",
        customer_address=None, purchase_order=None, currency={"status": "ambiguous"}, subtotal=None,
        tax_amount=None, total_amount=750.00, payment_terms=None, line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 6 inconsistent totals
def inv06():
    items = [("Plumbing repair - labour (hours)", 6, 95.0), ("Copper fittings", 1, 330.0)]
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "QuickFix Plumbing", 16, True); pg.t(50, 80, "17 Ash Avenue, Austin, TX 78701\nTax ID: 74-5550123")
    pg.rt(W - 50, 60, "INVOICE", 18, True); pg.rt(W - 50, 82, "No. QF-1043"); pg.rt(W - 50, 96, "Date: July 9, 2024")
    pg.t(50, 140, "Customer: Riverside Apartments HOA, 900 River Rd, Austin, TX 78703")
    y = pg.table(50, 180, ["Description", "Qty", "Rate", "Amount"], items_rows(items, us), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 10, "Subtotal: $900.00"); pg.rt(W - 50, y + 26, "Tax: $90.00")
    pg.rt(W - 50, y + 46, "TOTAL (USD): $1,100.00", 12, True)   # deliberate error: should be 990.00
    save(doc, "inv06_inconsistent_totals.pdf", dict(
        invoice_number="QF-1043", invoice_date="2024-07-09", due_date=None, vendor_name="QuickFix Plumbing",
        vendor_address="17 Ash Avenue, Austin, TX 78701", vendor_tax_id="74-5550123",
        customer_name="Riverside Apartments HOA", customer_address="900 River Rd, Austin, TX 78703",
        purchase_order=None, currency="USD", subtotal={"status": "inconsistent", "value": 900.00},
        tax_amount={"status": "inconsistent", "value": 90.00}, total_amount={"status": "inconsistent", "value": 1100.00},
        payment_terms=None, line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 7 scanned (image only)
def inv07():
    items = [("Lawn mowing service", 4, 45.0), ("Hedge trimming", 1, 80.0)]
    src = pymupdf.open(); pg = Page(src)
    pg.t(60, 70, "GreenThumb Gardening", 18, True); pg.t(60, 92, "5 Willow Close, Bristol BS8 1TH", 12)
    pg.t(60, 140, "INVOICE NO: GT-3381", 13, True); pg.t(60, 160, "DATE: 14 May 2024", 12)
    pg.t(60, 190, "BILL TO: Mrs Alice Morgan", 12)
    y = 230
    pg.t(60, y, "Description", 12, True); pg.t(330, y, "Qty", 12, True); pg.t(400, y, "Amount", 12, True)
    for d, q, p in items:
        y += 22; pg.t(60, y, d, 12); pg.t(330, y, str(q), 12); pg.t(400, y, f"£{us(q * p)}", 12)
    pg.t(330, y + 40, "TOTAL: £260.00", 14, True)
    pix = pg.p.get_pixmap(dpi=150)
    out = pymupdf.open(); op = out.new_page(width=W, height=H); op.insert_image(op.rect, pixmap=pix)
    save(out, "inv07_scanned.pdf", dict(
        invoice_number="GT-3381", invoice_date="2024-05-14", due_date=None, vendor_name="GreenThumb Gardening",
        vendor_address="5 Willow Close, Bristol BS8 1TH", vendor_tax_id=None, customer_name="Mrs Alice Morgan",
        customer_address=None, purchase_order=None, currency="GBP", subtotal=None, tax_amount=None,
        total_amount=260.00, payment_terms=None,
        line_items=[{"description": d, "quantity": q, "unit_price": None, "amount": q * p} for d, q, p in items]))


# ----------------------------------------------------------------------------- 8 label variants
def inv08():
    items = [("Cloud backup - 2 TB plan", 12, 19.99), ("Setup fee", 1, 49.0)]
    sub = sum(q * p for _, q, p in items); tax = round(sub * 0.05, 2)
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Seller", 9, True); pg.t(50, 74, "Nimbus Data Services Pte. Ltd.\n10 Anson Road #22-01\nSingapore 079903\nGST Reg: 201912345K")
    pg.t(320, 60, "Buyer", 9, True); pg.t(320, 74, "Coral Reef Travel Pte. Ltd.\n2 Orchard Turn\nSingapore 238801")
    pg.t(50, 160, f"Bill No. NDS/24/0457     Issued on: 2 September 2024     Pay by: 2 October 2024")
    pg.t(50, 176, "Your reference: CR-PO-2024-88")
    y = pg.table(50, 210, ["Particulars", "Units", "Rate SGD", "Value SGD"], items_rows(items, us), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 10, f"Taxable value: SGD {us(sub)}"); pg.rt(W - 50, y + 26, f"GST 5%: SGD {us(tax)}")
    pg.rt(W - 50, y + 46, f"Amount payable: SGD {us(sub + tax)}", 12, True)
    save(doc, "inv08_label_variants.pdf", dict(
        invoice_number="NDS/24/0457", invoice_date="2024-09-02", due_date="2024-10-02",
        vendor_name="Nimbus Data Services Pte. Ltd.", vendor_address="10 Anson Road #22-01, Singapore 079903",
        vendor_tax_id="201912345K", customer_name="Coral Reef Travel Pte. Ltd.",
        customer_address="2 Orchard Turn, Singapore 238801", purchase_order="CR-PO-2024-88", currency="SGD",
        subtotal=round(sub, 2), tax_amount=tax, total_amount=round(sub + tax, 2), payment_terms=None,
        line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 9 terms only, no due date
def inv09():
    items = [("Bookkeeping - Q2 2024", 1, 1200.0)]
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Ledger & Co Bookkeeping", 16, True); pg.t(50, 80, "9 Mill Street, Manchester M1 2HF\nVAT No. GB 123 4567 89")
    pg.t(50, 130, "Invoice reference: LC-2024-077"); pg.t(50, 144, "Invoice date: 1 July 2024")
    pg.t(50, 175, "To: Bloom Florists Ltd, 41 Deansgate, Manchester M3 2BW")
    y = pg.table(50, 210, ["Description", "Qty", "Fee", "Amount"], items_rows(items, us), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 10, "Net: GBP 1,200.00"); pg.rt(W - 50, y + 26, "VAT 20%: GBP 240.00")
    pg.rt(W - 50, y + 46, "Total: GBP 1,440.00", 12, True)
    pg.t(50, y + 90, "Payment due within 30 days of the invoice date.")
    save(doc, "inv09_terms_no_due_date.pdf", dict(
        invoice_number="LC-2024-077", invoice_date="2024-07-01", due_date=None, vendor_name="Ledger & Co Bookkeeping",
        vendor_address="9 Mill Street, Manchester M1 2HF", vendor_tax_id="GB 123 4567 89",
        customer_name="Bloom Florists Ltd", customer_address="41 Deansgate, Manchester M3 2BW", purchase_order=None,
        currency="GBP", subtotal=1200.00, tax_amount=240.00, total_amount=1440.00,
        payment_terms="Payment due within 30 days of the invoice date", line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 10 multi-page
def inv10():
    items = [(f"Consulting session {i} (hours)", 3, 120.0) for i in range(1, 25)]
    sub = sum(q * p for _, q, p in items); tax = round(sub * 0.1, 2)
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Apex Strategy Partners", 16, True); pg.t(50, 80, "1200 Pine Street, Seattle, WA 98101")
    pg.rt(W - 50, 60, "INVOICE APX-8890", 14, True); pg.rt(W - 50, 80, "Date: November 30, 2024"); pg.rt(W - 50, 94, "Due: December 30, 2024")
    pg.t(50, 130, "Bill to: Cascade Outdoor Gear Inc., 77 Lake Ave, Tacoma, WA 98402")
    pg.table(50, 170, ["Description", "Hours", "Rate", "Amount"], items_rows(items[:18], us), [0, 260, 330, 430], (2, 3))
    pg.t(50, H - 40, "Continued on next page", 8)
    pg2 = Page(doc)
    y = pg2.table(50, 70, ["Description", "Hours", "Rate", "Amount"], items_rows(items[18:], us), [0, 260, 330, 430], (2, 3))
    pg2.rt(W - 50, y + 10, f"Subtotal: ${us(sub)}"); pg2.rt(W - 50, y + 26, f"Tax (10%): ${us(tax)}")
    pg2.rt(W - 50, y + 46, f"Total due (USD): ${us(sub + tax)}", 12, True)
    save(doc, "inv10_multi_page.pdf", dict(
        invoice_number="APX-8890", invoice_date="2024-11-30", due_date="2024-12-30", vendor_name="Apex Strategy Partners",
        vendor_address="1200 Pine Street, Seattle, WA 98101", vendor_tax_id=None,
        customer_name="Cascade Outdoor Gear Inc.", customer_address="77 Lake Ave, Tacoma, WA 98402",
        purchase_order=None, currency="USD", subtotal=round(sub, 2), tax_amount=tax, total_amount=round(sub + tax, 2),
        payment_terms=None, line_items=gt_items(items)))


# ----------------------------------------------------------------------------- 11 receipt, no items
def inv11():
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(200, 70, "CORNER CAFE", 18, True); pg.t(190, 90, "3 Station Rd, Cambridge CB1 2JW")
    pg.t(200, 130, "Receipt / Invoice R-20240412-07"); pg.t(200, 146, "12 April 2024  13:42")
    pg.t(200, 180, "Catering for team lunch"); pg.t(200, 210, "TOTAL  GBP 186.40", 13, True)
    pg.t(200, 240, "VAT included (20%): GBP 31.07"); pg.t(200, 270, "Paid by card. Thank you!")
    save(doc, "inv11_receipt_no_items.pdf", dict(
        invoice_number="R-20240412-07", invoice_date="2024-04-12", due_date=None, vendor_name="Corner Cafe",
        vendor_address="3 Station Rd, Cambridge CB1 2JW", vendor_tax_id=None, customer_name=None,
        customer_address=None, purchase_order=None, currency="GBP", subtotal=None, tax_amount=31.07,
        total_amount=186.40, payment_terms=None, line_items=[]))


# ----------------------------------------------------------------------------- 12 India GST, lakh format
def inv12():
    items = [("Industrial water pump 5HP", 2, 45000.0), ("Installation and commissioning", 1, 15000.0), ("Spare impeller", 2, 2500.0)]
    sub = sum(q * p for _, q, p in items); gst = round(sub * 0.18, 2)
    doc = pymupdf.open(); pg = Page(doc)
    pg.t(50, 60, "Shakti Pumps & Motors Pvt. Ltd.", 15, True)
    pg.t(50, 80, "Plot 22, Peenya Industrial Area, Bengaluru 560058\nGSTIN: 29ABCDE1234F1Z5")
    pg.rt(W - 50, 60, "TAX INVOICE", 16, True); pg.rt(W - 50, 82, "Invoice No.: SPM/2023-24/1187"); pg.rt(W - 50, 96, "Date: 15-Jan-2024")
    pg.t(50, 140, "Billed To:", 10, True); pg.t(50, 155, "Sri Venkateswara Agro Farms\nSurvey No. 41, Hosur Road, Krishnagiri 635109")
    y = pg.table(50, 210, ["Description", "Qty", "Rate (INR)", "Amount (INR)"], items_rows(items, inr), [0, 260, 330, 430], (2, 3))
    pg.rt(W - 50, y + 10, f"Taxable Amount: Rs. {inr(sub)}"); pg.rt(W - 50, y + 26, f"IGST @ 18%: Rs. {inr(gst)}")
    pg.rt(W - 50, y + 46, f"Grand Total: Rs. {inr(sub + gst)}", 12, True)
    pg.t(50, y + 90, "Terms: 50% advance, balance within 15 days of delivery.")
    save(doc, "inv12_india_gst.pdf", dict(
        invoice_number="SPM/2023-24/1187", invoice_date="2024-01-15", due_date=None,
        vendor_name="Shakti Pumps & Motors Pvt. Ltd.", vendor_address="Plot 22, Peenya Industrial Area, Bengaluru 560058",
        vendor_tax_id="29ABCDE1234F1Z5", customer_name="Sri Venkateswara Agro Farms",
        customer_address="Survey No. 41, Hosur Road, Krishnagiri 635109", purchase_order=None, currency="INR",
        subtotal=round(sub, 2), tax_amount=gst, total_amount=round(sub + gst, 2),
        payment_terms="50% advance, balance within 15 days of delivery", line_items=gt_items(items)))


def edge_cases():
    EDGE.mkdir(exist_ok=True)
    (EDGE / "not_a_pdf.pdf").write_text("This is a text file with a .pdf extension", encoding="utf-8")
    (EDGE / "empty.pdf").write_bytes(b"")
    (EDGE / "invoice.docx").write_bytes(b"PK\x03\x04" + b"\x00" * 200)
    (EDGE / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    good = (HERE / "inv01_classic_usd.pdf").read_bytes()
    (EDGE / "truncated.pdf").write_bytes(good[: len(good) // 2])
    (EDGE / "garbage_after_header.pdf").write_bytes(b"%PDF-1.7\n" + bytes(range(256)) * 20)
    enc = pymupdf.open(HERE / "inv01_classic_usd.pdf")
    enc.save(EDGE / "encrypted.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret")
    blank = pymupdf.open(); blank.new_page(); blank.save(EDGE / "blank_page.pdf")


if __name__ == "__main__":
    for fn in (inv01, inv02, inv03, inv04, inv05, inv06, inv07, inv08, inv09, inv10, inv11, inv12):
        fn()
    (HERE / "ground_truth.json").write_text(json.dumps(GROUND_TRUTH, indent=2, ensure_ascii=False), encoding="utf-8")
    edge_cases()
    print(f"wrote {len(GROUND_TRUTH)} invoices + edge cases to {HERE}")
