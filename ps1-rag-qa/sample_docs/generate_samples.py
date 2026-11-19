"""Generate the sample corpus (fictional company "Aster Robotics").

The corpus is designed to exercise the hard cases:
- employee_handbook_2024.pdf vs leave_policy_update_2025.txt  -> CONFLICTING leave entitlement
- rx200_product_manual.pdf vs customer_support_faq.txt         -> DUPLICATED warranty paragraph
- multi-page PDFs with running headers/footers and page numbers -> header/footer removal
- edge_cases/                                                   -> invalid / empty / scanned inputs

Run:  python sample_docs/generate_samples.py
"""
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

HERE = Path(__file__).parent
EDGE = HERE / "edge_cases"

WARRANTY = ("The RX-200 is covered by a limited warranty of 2 years from the date of delivery. "
            "The warranty covers manufacturing defects in the drive system, lift mechanism and "
            "controller board. Batteries are covered for 12 months. Damage caused by operating the "
            "robot outside its rated payload or on wet floors is not covered.")

HANDBOOK = {
    "title": "Aster Robotics Employee Handbook",
    "header": "Aster Robotics - Employee Handbook 2024 - Internal",
    "sections": [
        ("Effective Date", ["Effective: January 1, 2024. This handbook applies to all full-time employees of Aster Robotics in all offices."]),
        ("1. Working Hours", [
            "Standard working hours are 9:00 to 17:30, Monday to Friday, with a 30 minute lunch break. "
            "Core collaboration hours, when every employee is expected to be reachable, are 10:00 to 15:00.",
            "Overtime must be approved in advance by your manager and is compensated as time off in lieu."]),
        ("2. Annual Leave", [
            "Full-time employees are entitled to 20 days of paid annual leave per calendar year, in addition to public holidays. "
            "Leave accrues monthly at a rate of 1.67 days per month.",
            "Leave requests of more than 5 consecutive days must be submitted at least 4 weeks in advance through the HR portal. "
            "Unused annual leave cannot be carried over into the next year."]),
        ("3. Sick Leave", [
            "Employees receive 10 days of paid sick leave per year. A medical certificate is required for absences longer than 3 consecutive days. "
            "Sick leave does not carry over."]),
        ("4. Remote Work", [
            "Employees may work remotely up to 2 days per week with manager approval. Remote days must be recorded in the team calendar. "
            "Employees working remotely must be available during core collaboration hours and must use the company VPN."]),
        ("5. Expense Reimbursement", [
            "Business expenses must be submitted through the finance portal within 30 days of the expense date, with an itemised receipt. "
            "Expenses submitted after 30 days require written approval from a department director.",
            "Reimbursements are paid with the next monthly payroll run after approval. Alcohol is not reimbursable except at approved client events."]),
        ("6. Learning and Development", [
            "Each employee has an annual learning budget of 1,500 USD for courses, certifications and conferences. "
            "Unused learning budget does not carry over. Requests above 500 USD require manager approval."]),
        ("7. Code of Conduct", [
            "Employees must treat colleagues, customers and partners with respect. Harassment and discrimination of any kind are not tolerated. "
            "Concerns can be raised with HR or anonymously through the ethics hotline."]),
    ],
}

MANUAL = {
    "title": "RX-200 Autonomous Warehouse Robot - Product Manual",
    "header": "RX-200 Product Manual v3.1",
    "sections": [
        ("1. Overview", [
            "The RX-200 is an autonomous mobile robot designed to move pallets and totes inside warehouses. "
            "It navigates using lidar and ceiling-mounted fiducial markers and integrates with warehouse management systems over a REST API."]),
        ("2. Technical Specifications", [
            "Maximum payload: 150 kg. Maximum speed: 1.8 metres per second. Dimensions: 980 x 650 x 340 mm. Weight without load: 85 kg.",
            "Battery: 48 V lithium iron phosphate. Typical runtime is 8 hours of continuous operation at 50 percent average payload. "
            "A full charge from 10 percent to 100 percent takes 2.5 hours on the standard dock; the fast dock reduces this to 1.5 hours.",
            "Operating temperature: 5 to 40 degrees Celsius. The robot is rated IP54 and must not be operated on wet floors."]),
        ("3. Installation", [
            "Before first use, map the facility with the mapping tool and place fiducial markers every 10 metres along main aisles. "
            "Charging docks must be installed on a level floor with at least 1 metre of clear space in front."]),
        ("4. Error Codes", [
            "E101 - Lidar obstruction: the lidar window is dirty or blocked. Clean the window with a dry microfibre cloth and restart the navigation module.",
            "E204 - Battery overheating: the battery temperature exceeded 55 degrees Celsius. The robot stops and returns to the dock. "
            "Let it cool for 30 minutes; if the error repeats, contact support.",
            "E307 - Payload imbalance: the load sensors detected an uneven or excessive load. Rebalance the load or reduce it below 150 kg.",
            "E410 - Localisation lost: the robot cannot match its position to the map. Move it manually to a known marker and press Resume."]),
        ("5. Maintenance Schedule", [
            "Weekly: clean lidar window and wheel treads. Monthly: inspect drive wheels for wear and check lift actuator for noise. "
            "Every 6 months: replace drive wheel tyres if tread depth is under 2 mm, and have a certified technician run the safety diagnostic."]),
        ("6. Warranty", [WARRANTY]),
        ("7. Safety", [
            "The RX-200 stops automatically when a person is detected within 0.5 metres of its path. The red emergency stop button on each side "
            "cuts power to the drive motors immediately. Only trained operators may override safety zones."]),
    ],
}

TRAVEL = {
    "title": "Aster Robotics Travel Policy",
    "header": "Travel Policy - Finance Department",
    "sections": [
        ("Scope", ["Last updated: March 15, 2024. This policy covers all business travel booked by Aster Robotics employees."]),
        ("Flights", [
            "Economy class is required for flights under 6 hours. Premium economy is allowed for flights of 6 hours or longer. "
            "Business class requires approval from the CFO. Flights must be booked at least 14 days in advance when possible."]),
        ("Hotels", [
            "The nightly hotel cap is 180 USD for domestic travel and 250 USD for international travel, excluding taxes. "
            "Employees should use the corporate booking tool to access negotiated rates."]),
        ("Per Diem", [
            "The daily meal allowance (per diem) is 60 USD for domestic trips and 85 USD for international trips. "
            "Per diem is not paid for days when meals are provided by a client or a conference."]),
        ("Ground Transport", [
            "Public transport or ride-sharing should be used where practical. Rental cars require manager approval and must be compact or mid-size."]),
    ],
}

LEAVE_UPDATE_TXT = """Aster Robotics - HR Policy Update: Annual Leave
Effective: January 1, 2025

This update replaces section 2 (Annual Leave) of the 2024 Employee Handbook.

Annual Leave Entitlement
From January 1, 2025, full-time employees are entitled to 24 days of paid annual leave per calendar year, in addition to public holidays. Part-time employees receive a pro-rated entitlement.

Carry-Over
Employees may carry over up to 5 unused annual leave days into the next calendar year. Carried-over days must be used by March 31 of the following year, otherwise they expire.

Requesting Leave
The 4-week advance notice rule for leave longer than 5 consecutive days remains unchanged.
"""

SECURITY_TXT = """Aster Robotics - IT Security Policy
Last updated: June 1, 2024

Passwords
All accounts must use passwords of at least 14 characters. Passwords must be changed every 180 days and must not be reused within the last 10 passwords. Use of the company password manager is mandatory.

Multi-Factor Authentication
Multi-factor authentication (MFA) is required for email, VPN, the HR portal and all production systems. Hardware security keys are issued to engineers with production access.

Devices
Company laptops must have full-disk encryption enabled and must lock automatically after 5 minutes of inactivity. Personal devices may access email only through the managed mobile app.

Reporting Incidents
Suspected phishing emails must be reported using the "Report Phishing" button or by forwarding them to security@aster-robotics.example. Lost or stolen devices must be reported to the IT helpdesk within 24 hours.

Data Classification
Data is classified as Public, Internal, Confidential or Restricted. Restricted data, such as customer contracts and employee records, must never be stored on personal devices or shared through personal email.
"""

FAQ_TXT = f"""RX-200 Customer Support FAQ
Last updated: August 20, 2024

What are your support hours?
Our support team is available Monday to Friday, 08:00 to 20:00 CET, by phone and through the customer portal. Critical incidents (robot fleet down) are handled 24/7 for customers with a Premium support plan.

What warranty does the RX-200 have?
{WARRANTY}

Can I return an RX-200?
Robots can be returned within 30 days of delivery if they are unused and in the original packaging. A 15 percent restocking fee applies.

How do I update the robot software?
Software updates are delivered over the air through the fleet manager. Updates are installed when the robot is docked and idle. You can schedule a maintenance window in fleet manager settings.

How many robots can one fleet manager control?
A single fleet manager instance supports up to 50 RX-200 robots. Larger deployments use multiple fleet manager instances with a shared map.
"""


def write_pdf(spec: dict, path: Path):
    doc = fitz.open()
    W, H = fitz.paper_size("a4")
    margin, line_h = 60, 15
    page = None
    y = 0
    pages = []

    def new_page():
        nonlocal page, y
        page = doc.new_page(width=W, height=H)
        pages.append(page)
        page.insert_text((margin, 35), spec["header"], fontsize=8, color=(0.4, 0.4, 0.4))
        y = 70

    def write_wrapped(text, size, bold=False):
        nonlocal y
        font = "helv" if not bold else "hebo"
        words, line = text.split(), ""
        max_w = W - 2 * margin
        for w in words + [None]:
            cand = (line + " " + w).strip() if w else line
            if w and fitz.get_text_length(cand, fontname=font, fontsize=size) <= max_w:
                line = cand
                continue
            if y > H - 80:
                new_page()
            page.insert_text((margin, y), line, fontsize=size, fontname=font)
            y += line_h if size < 12 else line_h + 4
            line = w or ""

    new_page()
    write_wrapped(spec["title"], 15, bold=True)
    y += 10
    for n_sec, (heading, paras) in enumerate(spec["sections"]):
        # page break every 3 sections so documents span several pages (exercises header/footer removal)
        if y > H - 140 or (n_sec and n_sec % 3 == 0):
            new_page()
        y += 6
        write_wrapped(heading, 11.5, bold=True)
        y += 2
        for para in paras:
            write_wrapped(para, 10)
            y += 8
    n = doc.page_count
    for i, p in enumerate(doc, start=1):
        p.insert_text((W / 2 - 25, H - 30), f"Page {i} of {n}", fontsize=8, color=(0.4, 0.4, 0.4))
    doc.save(path)


def write_edge_cases():
    EDGE.mkdir(exist_ok=True)
    (EDGE / "not_really_a_pdf.pdf").write_text("This is plain text pretending to be a PDF.", encoding="utf-8")
    (EDGE / "empty.txt").write_text("", encoding="utf-8")
    (EDGE / "spreadsheet.xlsx").write_bytes(b"PK\x03\x04 fake xlsx content")
    # Image-only "scanned" PDF: render text to an image and embed only the image.
    src = fitz.open()
    pg = src.new_page()
    pg.insert_text((72, 100), "Scanned memo: the office closes at 18:00 on Fridays.", fontsize=12)
    pix = pg.get_pixmap(dpi=100)
    out = fitz.open()
    op = out.new_page()
    op.insert_image(op.rect, pixmap=pix)
    out.save(EDGE / "scanned_memo.pdf")
    # Truncated / corrupted PDF
    good = (HERE / "travel_policy.pdf").read_bytes()
    (EDGE / "corrupted.pdf").write_bytes(good[: len(good) // 3])


if __name__ == "__main__":
    write_pdf(HANDBOOK, HERE / "employee_handbook_2024.pdf")
    write_pdf(MANUAL, HERE / "rx200_product_manual.pdf")
    write_pdf(TRAVEL, HERE / "travel_policy.pdf")
    (HERE / "leave_policy_update_2025.txt").write_text(LEAVE_UPDATE_TXT, encoding="utf-8")
    (HERE / "it_security_policy.txt").write_text(SECURITY_TXT, encoding="utf-8")
    (HERE / "customer_support_faq.txt").write_text(FAQ_TXT, encoding="utf-8")
    write_edge_cases()
    print("Sample documents written to", HERE)
