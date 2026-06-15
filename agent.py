from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import traceback
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

def convert_to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_to_native(i) for i in obj]
    return obj


def calculate_gst_analytics(previous_df, current_df):
    # Sales
    prev_sales = previous_df["taxable_value"].sum() if not previous_df.empty else 0
    curr_sales = current_df["taxable_value"].sum() if not current_df.empty else 0

    sales_growth_pct = (((curr_sales - prev_sales) / prev_sales) * 100 if prev_sales > 0 else 0)

    # Invoice Count
    prev_invoice_count = len(previous_df)
    curr_invoice_count = len(current_df)

    invoice_growth_pct = (((curr_invoice_count - prev_invoice_count) / prev_invoice_count) * 100 if prev_invoice_count > 0 else 0)

    # Customers
    prev_customers = set(previous_df["customer_name"].dropna()) if "customer_name" in previous_df else set()
    curr_customers = set(current_df["customer_name"].dropna()) if "customer_name" in current_df else set()

    new_customers = curr_customers - prev_customers
    lost_customers = prev_customers - curr_customers
    repeat_customers = prev_customers & curr_customers

    repeat_ratio = (len(repeat_customers) / len(curr_customers) * 100 if len(curr_customers) > 0 else 0)
    retention_rate = (len(repeat_customers) / len(prev_customers) * 100 if len(prev_customers) > 0 else 0)
    churn_rate = (len(lost_customers) / len(prev_customers) * 100 if len(prev_customers) > 0 else 0)
    
    avg_rev_per_customer = curr_sales / len(curr_customers) if len(curr_customers) > 0 else 0
    purchase_freq = curr_invoice_count / len(curr_customers) if len(curr_customers) > 0 else 0
    daily_run_rate = curr_sales / 30

    # Top Customers
    if "customer_name" in current_df and not current_df.empty:
        top_customers = current_df.groupby("customer_name")["taxable_value"].sum().sort_values(ascending=False).head(10).reset_index()
    else:
        top_customers = pd.DataFrame(columns=["customer_name", "taxable_value"])

    # Customer Growth & Risk
    if "customer_name" in previous_df and not previous_df.empty:
        prev_cust = previous_df.groupby("customer_name")["taxable_value"].sum()
    else:
        prev_cust = pd.Series(dtype=float)

    if "customer_name" in current_df and not current_df.empty:
        curr_cust = current_df.groupby("customer_name")["taxable_value"].sum()
    else:
        curr_cust = pd.Series(dtype=float)

    growth_rows = []
    risk_drop_20_count = 0
    revenue_at_risk = 0

    for customer in curr_cust.index:
        prev_val = prev_cust.get(customer, 0)
        curr_val = curr_cust.get(customer, 0)
        if prev_val > 0:
            growth = ((curr_val - prev_val) / prev_val) * 100
            growth_rows.append({"customer": customer, "growth_pct": round(growth, 2), "current_sales": curr_val})
            if growth < -20:
                risk_drop_20_count += 1
                revenue_at_risk += (prev_val - curr_val)

    fastest_growing = pd.DataFrame(growth_rows).sort_values("growth_pct", ascending=False).head(10) if growth_rows else pd.DataFrame(columns=["customer", "growth_pct", "current_sales"])

    # Average Order Value
    aov = (curr_sales / curr_invoice_count if curr_invoice_count > 0 else 0)

    # Customer Concentration
    top5_sales = top_customers["taxable_value"].head(5).sum() if not top_customers.empty else 0
    customer_concentration = (top5_sales / curr_sales * 100 if curr_sales > 0 else 0)

    return {
        "sales_growth_pct": round(sales_growth_pct, 2),
        "invoice_growth_pct": round(invoice_growth_pct, 2),
        "new_customers": len(new_customers),
        "lost_customers": len(lost_customers),
        "repeat_customers": len(repeat_customers),
        "repeat_ratio": round(repeat_ratio, 2),
        "retention_rate": round(retention_rate, 2),
        "churn_rate": round(churn_rate, 2),
        "avg_rev_per_customer": round(avg_rev_per_customer, 2),
        "purchase_freq": round(purchase_freq, 2),
        "daily_run_rate": round(daily_run_rate, 2),
        "risk_drop_20_count": risk_drop_20_count,
        "revenue_at_risk": round(revenue_at_risk, 2),
        "avg_order_value": round(aov, 2),
        "customer_concentration": round(customer_concentration, 2),
        "top_customers": top_customers.to_dict("records"),
        "fastest_growing": fastest_growing.to_dict("records")
    }

app = Flask(__name__)
app.secret_key = 'super_secret_production_key'

def init_db():
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL,
                  upload_count INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

init_db()

# Using global variable for simplicity in this prototype
global_state = {
    "df": None,
    "stats": None
}

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.sarvam.ai/v1"
)
MODEL_NAME = "sarvam-30b"

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = sqlite3.connect('app.db')
        c = conn.cursor()
        c.execute("SELECT id, password FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()
        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        hashed = generate_password_hash(password)
        try:
            conn = sqlite3.connect('app.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already registered")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop('user_id', None)
    return redirect(url_for("login"))

@app.route("/")
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

@app.route("/api/upload", methods=["POST"])
def upload_files():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute("SELECT upload_count FROM users WHERE id=?", (session['user_id'],))
    row = c.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "User not found"}), 401
        
    upload_count = row[0]
    
    if upload_count >= 3:
        conn.close()
        return jsonify({"error": "Upload limit reached. You can only analyze data 3 times on this account."}), 403
        
    c.execute("UPDATE users SET upload_count = upload_count + 1 WHERE id=?", (session['user_id'],))
    conn.commit()
    conn.close()

    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    all_data = []
    
    try:
        for file in files:
            file_dfs = []
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                df['sheet_name'] = 'B2B'
                file_dfs.append(df)
            else:
                xls = pd.read_excel(file, sheet_name=None)
                for sheet_name, df in xls.items():
                    if df.empty: continue
                    df['sheet_name'] = str(sheet_name).upper().strip()
                    file_dfs.append(df)
            
            if not file_dfs: continue
            
            month_df = pd.concat(file_dfs, ignore_index=True)
            month_df['Source_File'] = file.filename
            
            # Clean column names strictly of leading/trailing spaces
            month_df.columns = [str(col).strip() for col in month_df.columns]
            
            # B2B / B2C Detection via sheet name OR Type column
            if 'Type' in month_df.columns:
                month_df['type'] = month_df['Type'].fillna('Unknown')
            else:
                def get_type(s):
                    if 'B2B' in s: return 'B2B'
                    if 'B2CS' in s or 'B2C' in s: return 'B2C'
                    if 'EXP' in s: return 'Exports'
                    if 'CDN' in s: return 'Credit Notes'
                    return s
                month_df['type'] = month_df['sheet_name'].apply(get_type)
            
            # Customer Name Mapping
            if 'Receiver Name' in month_df.columns: month_df['customer_name'] = month_df['Receiver Name']
            elif 'Customer Name' in month_df.columns: month_df['customer_name'] = month_df['Customer Name']
            elif 'Party Name' in month_df.columns: month_df['customer_name'] = month_df['Party Name']
            else: month_df['customer_name'] = 'Unknown'
            
            # State Mapping
            if 'Place of Supply' in month_df.columns: month_df['state'] = month_df['Place of Supply']
            elif 'State' in month_df.columns: month_df['state'] = month_df['State']
            elif 'POS' in month_df.columns: month_df['state'] = month_df['POS']
            else: month_df['state'] = 'Unknown'
            
            # Taxable Value Mapping
            tv_col = next((c for c in month_df.columns if 'Taxable' in c and 'Value' in c), None)
            if not tv_col: tv_col = next((c for c in month_df.columns if 'taxable' in c.lower()), None)
            if not tv_col: tv_col = next((c for c in month_df.columns if 'value' in c.lower() or 'amount' in c.lower()), None)
            
            if tv_col:
                month_df['taxable_value'] = pd.to_numeric(month_df[tv_col], errors='coerce').fillna(0)
            else:
                month_df['taxable_value'] = 0

            # Map remaining GSTR-1 fields to lowercase standard names for quality checks
            if 'GSTIN/UIN' in month_df.columns: month_df['gstin'] = month_df['GSTIN/UIN']
            if 'Invoice Number' in month_df.columns: month_df['invoice_number'] = month_df['Invoice Number']
            elif 'Invoice No' in month_df.columns: month_df['invoice_number'] = month_df['Invoice No']
            if 'Invoice Date' in month_df.columns: month_df['invoice_date'] = month_df['Invoice Date']
            
            all_data.append(month_df)
            
        if not all_data:
            return jsonify({"error": "Failed to read files"}), 400
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure fallback for gstin and dates so quality checks don't crash
        if 'gstin' not in combined_df.columns: combined_df['gstin'] = np.nan
        if 'invoice_date' not in combined_df.columns: combined_df['invoice_date'] = np.nan
        if 'invoice_number' not in combined_df.columns: combined_df['invoice_number'] = np.nan

        if 'invoice_date' in combined_df.columns and combined_df['invoice_date'].notna().any():
            combined_df['parsed_date'] = pd.to_datetime(combined_df['invoice_date'], format='mixed', dayfirst=True, errors='coerce')
            combined_df['year_month'] = combined_df['parsed_date'].dt.strftime('%b %Y')
            
            # Sort months chronologically
            valid_dates = combined_df.dropna(subset=['parsed_date'])
            if not valid_dates.empty:
                month_order = valid_dates.groupby('year_month')['parsed_date'].min().sort_values().index.tolist()
            else:
                month_order = []
                
            if len(month_order) >= 2:
                previous_df = combined_df[combined_df['year_month'] == month_order[-2]]
                current_df = combined_df[combined_df['year_month'] == month_order[-1]]
            elif len(month_order) == 1:
                previous_df = pd.DataFrame(columns=combined_df.columns)
                current_df = combined_df[combined_df['year_month'] == month_order[0]]
            else:
                previous_df = all_data[-2] if len(all_data) >= 2 else pd.DataFrame(columns=combined_df.columns)
                current_df = all_data[-1] if len(all_data) >= 2 else all_data[0]
                
            trend_data = [{"Month": m, "Sales": combined_df[combined_df['year_month'] == m]["taxable_value"].sum()} for m in month_order]
        else:
            previous_df = all_data[-2] if len(all_data) >= 2 else pd.DataFrame(columns=combined_df.columns)
            current_df = all_data[-1] if len(all_data) >= 2 else all_data[0]
            trend_data = []

        analytics = calculate_gst_analytics(previous_df, current_df)
        
        # B2B / B2C Extended Analysis
        b2b_df = current_df[current_df["type"].astype(str).str.contains("B2B", case=False, na=False)]
        b2c_df = current_df[current_df["type"].astype(str).str.contains("B2C", case=False, na=False)]
        
        b2b_sales = b2b_df["taxable_value"].sum()
        b2c_sales = b2c_df["taxable_value"].sum()
        total_sales = b2b_sales + b2c_sales
        b2b_ratio = (b2b_sales / total_sales * 100) if total_sales > 0 else 0
        b2c_ratio = (b2c_sales / total_sales * 100) if total_sales > 0 else 0
        
        avg_b2b_invoice = b2b_sales / len(b2b_df) if len(b2b_df) > 0 else 0
        avg_b2c_invoice = b2c_sales / len(b2c_df) if len(b2c_df) > 0 else 0
        
        prev_b2b_sales = previous_df[previous_df["type"].astype(str).str.contains("B2B", case=False, na=False)]["taxable_value"].sum() if 'type' in previous_df.columns else 0
        prev_b2c_sales = previous_df[previous_df["type"].astype(str).str.contains("B2C", case=False, na=False)]["taxable_value"].sum() if 'type' in previous_df.columns else 0
        
        b2b_growth_pct = ((b2b_sales - prev_b2b_sales) / prev_b2b_sales * 100) if prev_b2b_sales > 0 else 0
        b2c_growth_pct = ((b2c_sales - prev_b2c_sales) / prev_b2c_sales * 100) if prev_b2c_sales > 0 else 0

        # State Analysis
        state_analysis = current_df.groupby("state")["taxable_value"].sum().sort_values(ascending=False).reset_index()

        # Ensure we don't crash on missing columns
        has_gstin = 'gstin' in combined_df.columns
        has_type = 'type' in combined_df.columns
        has_date = 'invoice_date' in combined_df.columns
        has_inv = 'invoice_number' in combined_df.columns

        quality_issues = {
            "missing_gstin": int(combined_df[(combined_df['type'].astype(str).str.upper() == 'B2B') & combined_df['gstin'].isna()].shape[0]) if has_gstin and has_type else 0,
            "missing_date": int(combined_df['invoice_date'].isna().sum()) if has_date else 0,
            "duplicates": int(combined_df.duplicated(subset=['invoice_number']).sum()) if has_inv else 0
        }

        # Final Dashboard KPIs
        dashboard_kpis = {
            "Total Sales": total_sales,
            "Sales Growth %": analytics["sales_growth_pct"],
            "Invoice Growth %": analytics["invoice_growth_pct"],
            "Daily Sales Run Rate": analytics["daily_run_rate"],
            "New Customers": analytics["new_customers"],
            "Lost Customers": analytics["lost_customers"],
            "Repeat Customers": analytics["repeat_customers"],
            "Repeat Ratio": analytics["repeat_ratio"],
            "Retention Rate": analytics["retention_rate"],
            "Churn Rate": analytics["churn_rate"],
            "Avg Rev per Customer": analytics["avg_rev_per_customer"],
            "Customer Purchase Freq": analytics["purchase_freq"],
            "Risk Drop >20% Count": analytics["risk_drop_20_count"],
            "Revenue at Risk": analytics["revenue_at_risk"],
            "Average Order Value": analytics["avg_order_value"],
            "B2B Ratio": round(b2b_ratio, 2),
            "B2C Ratio": round(b2c_ratio, 2),
            "B2B Growth %": round(b2b_growth_pct, 2),
            "B2C Growth %": round(b2c_growth_pct, 2),
            "Avg B2B Invoice": round(avg_b2b_invoice, 2),
            "Avg B2C Invoice": round(avg_b2c_invoice, 2),
            "Customer Concentration %": analytics["customer_concentration"],
            "Top States": state_analysis.to_dict("records"),
            "GST Errors": quality_issues["missing_gstin"] + quality_issues["missing_date"],
            "Duplicate Invoices": quality_issues["duplicates"]
        }

        stats = dashboard_kpis.copy()
        # Keep essential old keys for backward compatibility in the chat feature
        stats["taxable_value"] = total_sales
        stats["total_invoices"] = len(combined_df)
        stats["total_gst"] = total_sales * 0.18
        stats["quality_issues"] = quality_issues
        stats["categories"] = {"B2B": b2b_sales, "B2C": b2c_sales, "Exports": 0, "Credit Notes": 0}
        
        global_state['df'] = combined_df
        global_state['stats'] = stats
        
        # Build HTML tables manually for immediate feedback based on real computed data
        top_cust_html = "".join([f"<tr><td>{c['customer_name']}</td><td>₹{c['taxable_value']:,.2f}</td></tr>" for c in analytics["top_customers"][:5]])
        top_states_html = "".join([f"<tr><td>{s['state']}</td><td>₹{s['taxable_value']:,.2f}</td></tr>" for s in state_analysis.to_dict("records")[:5]])
        trend_html = "".join([f"<tr><td>{t['Month']}</td><td class='text-end'>₹{t['Sales']:,.2f}</td></tr>" for t in trend_data]) if trend_data else "<tr><td colspan='2'>No date data available for trend</td></tr>"

        stats['insights'] = f"""
            <h6 class="text-dark fw-bold border-bottom pb-2">Executive Dashboard</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <tbody>
                        <tr><th>Total Sales (Current Month)</th><td>₹{total_sales:,.2f}</td></tr>
                        <tr><th>Sales Growth %</th><td>{analytics['sales_growth_pct']}%</td></tr>
                        <tr><th>Invoice Growth %</th><td>{analytics['invoice_growth_pct']}%</td></tr>
                        <tr><th>Daily Sales Run Rate</th><td>₹{analytics['daily_run_rate']:,.2f}</td></tr>
                        <tr><th>Average Order Value</th><td>₹{analytics['avg_order_value']:,.2f}</td></tr>
                    </tbody>
                </table>
            </div>
            
            <h6 class="text-dark fw-bold border-bottom pb-2 mt-4">Monthly Sales Trend</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <thead><tr><th>Month</th><th class="text-end">Sales Value</th></tr></thead>
                    <tbody>{trend_html}</tbody>
                </table>
            </div>
            
            <h6 class="text-dark fw-bold border-bottom pb-2 mt-4">Customer Analytics Deep Dive</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <tbody>
                        <tr><th>New Customers</th><td>{analytics['new_customers']}</td></tr>
                        <tr><th>Lost Customers</th><td>{analytics['lost_customers']}</td></tr>
                        <tr><th>Repeat Customers</th><td>{analytics['repeat_customers']}</td></tr>
                        <tr><th>Retention Rate</th><td>{analytics['retention_rate']}%</td></tr>
                        <tr><th>Churn Rate</th><td>{analytics['churn_rate']}%</td></tr>
                        <tr><th>Avg Revenue per Customer</th><td>₹{analytics['avg_rev_per_customer']:,.2f}</td></tr>
                        <tr><th>Purchase Frequency</th><td>{analytics['purchase_freq']} invoices/customer</td></tr>
                        <tr><th>Customer Concentration (Top 5)</th><td>{analytics['customer_concentration']}%</td></tr>
                    </tbody>
                </table>
            </div>
            
            <h6 class="text-dark fw-bold border-bottom pb-2 mt-4">B2B vs B2C deep dive</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <tbody>
                        <tr><th>B2B / B2C Ratio</th><td>{b2b_ratio:.1f}% / {b2c_ratio:.1f}%</td></tr>
                        <tr><th>B2B Growth %</th><td>{b2b_growth_pct:.1f}%</td></tr>
                        <tr><th>B2C Growth %</th><td>{b2c_growth_pct:.1f}%</td></tr>
                        <tr><th>Avg B2B Invoice Value</th><td>₹{avg_b2b_invoice:,.2f}</td></tr>
                        <tr><th>Avg B2C Invoice Value</th><td>₹{avg_b2c_invoice:,.2f}</td></tr>
                    </tbody>
                </table>
            </div>
            
            <h6 class="text-dark fw-bold border-bottom pb-2 mt-4">Risk & Opportunity Indicators</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <tbody>
                        <tr><th>Customers with Sales Drop > 20%</th><td class="text-danger fw-bold">{analytics['risk_drop_20_count']}</td></tr>
                        <tr><th>Revenue at Risk (Lost to Decline)</th><td class="text-danger fw-bold">₹{analytics['revenue_at_risk']:,.2f}</td></tr>
                        <tr><th>Inactive Customers (Churn)</th><td>{analytics['lost_customers']}</td></tr>
                    </tbody>
                </table>
            </div>

            <h6 class="text-dark fw-bold border-bottom pb-2 mt-4">Top 5 Customers by Sales</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <thead><tr><th>Customer Name</th><th>Sales</th></tr></thead>
                    <tbody>{top_cust_html if top_cust_html else "<tr><td colspan='2'>No data</td></tr>"}</tbody>
                </table>
            </div>

            <h6 class="text-dark fw-bold border-bottom pb-2 mt-4">State Analysis (Top 5)</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <thead><tr><th>State</th><th>Sales</th></tr></thead>
                    <tbody>{top_states_html if top_states_html else "<tr><td colspan='2'>No data</td></tr>"}</tbody>
                </table>
            </div>

            <h6 class="text-dark fw-bold border-bottom pb-2 mt-4">Data Quality & Risk Indicators</h6>
            <div class="table-responsive">
                <table class="table table-bordered table-striped">
                    <tbody>
                        <tr><th>Duplicate Invoices</th><td>{dashboard_kpis['Duplicate Invoices']}</td></tr>
                    </tbody>
                </table>
            </div>
        """

        return jsonify({"success": True, "stats": convert_to_native(stats)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_q = data.get("question")
    
    if global_state['df'] is None or global_state['stats'] is None:
        return jsonify({"error": "Please upload data first."}), 400
        
    stats = global_state['stats']
    
    chat_prompt = f"""
    You are an expert Sales & Financial Analyst AI.
    The user has uploaded their GSTR-1 data. Here are their actual computed metrics for the current period (with comparisons to the previous period):
    
    - Total Sales: ₹{stats.get('Total Sales', 0):,.2f}
    - Sales Growth: {stats.get('Sales Growth %', 0)}%
    - B2B vs B2C Ratio: {stats.get('B2B Ratio', 0)}% B2B / {stats.get('B2C Ratio', 0)}% B2C
    - Customer Churn Rate: {stats.get('Churn Rate', 0)}%
    - Customer Retention Rate: {stats.get('Retention Rate', 0)}%
    - New Customers: {stats.get('New Customers', 0)} | Lost Customers (Churn): {stats.get('Lost Customers', 0)}
    - Average Order Value (AOV): ₹{stats.get('Average Order Value', 0):,.2f}
    - Average Revenue per Customer: ₹{stats.get('Avg Rev per Customer', 0):,.2f}
    - Revenue at Risk (from declining clients): ₹{stats.get('Revenue at Risk', 0):,.2f}
    - Top Performing States: {', '.join([s.get('state', 'Unknown') for s in stats.get('Top States', [])[:5]])}
    
    If the user asks for strategies or suggestions (e.g. "how can I increase sales growth?"), formulate tailored, specific business strategies based on the metrics.
    
    CRITICAL RULE: Your ENTIRE response MUST be EXACTLY 5 sentences long (roughly 5 lines). Be punchy, highly analytical, and get straight to the point. Do not exceed this limit or your answer will be cut off.
    
    User question: "{user_q}"
    """
    
    try:
        chat_resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful GST data assistant."},
                {"role": "user", "content": chat_prompt}
            ]
        )
        content = chat_resp.choices[0].message.content
        if not content:
            raise Exception("API returned empty content")
        return jsonify({"answer": content})
    except Exception as e:
        q = user_q.lower()
        stats = global_state['stats']
        
        if any(word in q for word in ['total', 'revenue', 'taxable']):
            ans = f"The total taxable value is ₹{stats['taxable_value']:,.2f}."
        elif any(word in q for word in ['gst', 'liability', 'tax']):
            ans = f"The total GST liability is ₹{stats['total_gst']:,.2f}."
        elif any(word in q for word in ['invoice', 'count', 'how many']):
            ans = f"There are a total of {stats['total_invoices']} invoices in the uploaded data."
        elif 'export' in q:
            ans = f"The total value of export sales is ₹{stats['categories'].get('Exports', 0):,.2f}."
        elif 'b2b' in q:
            ans = f"B2B sales amount to ₹{stats['categories'].get('B2B', 0):,.2f}."
        elif 'b2c' in q:
            ans = f"B2C sales amount to ₹{stats['categories'].get('B2C', 0):,.2f}."
        elif any(word in q for word in ['missing', 'invalid', 'gstin']):
            ans = f"There are {stats['quality_issues'].get('missing_gstin', 0)} invoices with missing GSTINs."
        elif 'duplicate' in q:
            ans = f"I found {stats['quality_issues'].get('duplicates', 0)} duplicate invoices."
        elif any(word in q for word in ['increase', 'grow', 'strategy', 'churn', 'retention', 'improve']):
            ans = f"Your current sales growth is {stats.get('Sales Growth %', 0)}% and your churn rate is {stats.get('Churn Rate', 0)}%. You lost {stats.get('Lost Customers', 0)} customers, putting ₹{stats.get('Revenue at Risk', 0):,.2f} at direct risk. To increase growth immediately, launch a targeted reactivation campaign to win back those {stats.get('Lost Customers', 0)} specific clients. Secondly, your B2B ratio is {stats.get('B2B Ratio', 0):.1f}%, so focus your proactive retention check-ins exclusively on your enterprise tier. Finally, your Average Order Value is ₹{stats.get('Average Order Value', 0):,.2f}, which presents a strong cross-selling opportunity for your existing base."
        else:
            ans = f"I've analyzed {len(global_state['df'])} rows of your data. The total taxable value calculated is ₹{stats['taxable_value']:,.2f}. Try asking about 'revenue', 'GST', 'growth', 'B2B', 'exports', or 'duplicates'!"
            
        return jsonify({"answer": ans + " (Note: Smart rule-based response due to AI API unavailability)."})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
