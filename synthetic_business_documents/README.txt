SYNTHETIC BUSINESS DOCUMENT DATASET
====================================

Dataset Summary
---------------
Total Documents: 50
Document Types: 5 (Invoice, Purchase Order, Resume, Contract, Offer Letter)
Documents per Type: 10
Language: English
Business Context: Indian enterprises
Currency: INR (Indian Rupees)

Difficulty Distribution
-----------------------
Easy: 18 documents (36%)
Medium: 22 documents (44%)
Hard: 10 documents (20%)

Noise Level Distribution
-------------------------
None: 33 documents (66%)
Low: 11 documents (22%)
Medium: 6 documents (12%)

Document Categories
-------------------

1. INVOICES (10 documents)
   - Service invoices, product invoices, professional fees
   - Formats: formal letterhead, email-based, informal
   - Industries: IT services, logistics, creative agencies, pharmaceuticals, legal, facilities, education, manufacturing

2. PURCHASE ORDERS (10 documents)
   - Product procurement, service contracts, equipment orders
   - Formats: formal PO, email confirmation, internal memo
   - Industries: Retail, construction, healthcare, textiles, IT, furniture, packaging, software, pharmaceuticals, education

3. RESUMES (10 documents)
   - Technical, marketing, finance, HR, operations, healthcare, engineering, design, content
   - Formats: traditional, modern, casual/informal
   - Experience levels: 4-10 years

4. CONTRACTS (10 documents)
   - Service agreements, distribution, consulting, leases, franchises, partnerships, software licenses, manufacturing
   - Formats: formal legal, business letters, email-based
   - Term lengths: 1-5 years

5. OFFER LETTERS (10 documents)
   - Tech, consulting, startups, manufacturing, retail, healthcare, logistics, education
   - Formats: formal HR letterhead, email-based, casual startup style
   - Salary ranges: 4-24 LPA

Key Features
------------
- No explicit document type labels in body text
- Realistic Indian business names, locations, GST numbers
- Professional tone with varying formality levels
- Implicit signals for document classification
- Mixed structured and unstructured formats
- Representative of actual enterprise documents

Use Cases
---------
- Training document classification models
- Testing AI-powered routing systems
- Evaluating LLM document understanding
- Building document processing pipelines
- Benchmarking extraction accuracy

File Structure
--------------
documents/
  ├── invoices/           (10 files)
  ├── purchase_orders/    (10 files)
  ├── resumes/           (10 files)
  ├── contracts/         (10 files)
  └── offer_letters/     (10 files)

document_index.csv - Complete metadata index

Notes
-----
- All data is synthetic and fictional
- No real personal or company information used
- Documents range from 200-350 words
- Plain text format (.txt)
- Created for AI testing and development purposes
