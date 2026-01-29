"""
Module for generating a comprehensive HTML report of the entire Invisible Variables Engine pipeline.

Aggregates:
- Validation results (Proxy, Retraining)
- Invisible Variable details (Explanations)
- Visualizations (Base64 encoded)

Produce a standalone HTML file sharing the "Detective's Report" on model failure modes.
"""

import os
import pandas as pd
import json
import base64
from jinja2 import Template

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Invisible Variables Engine - Forensic ML Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f9f9f9; }
        h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
        .card { background: #f8f9fa; border-left: 5px solid #3498db; padding: 15px; margin-bottom: 20px; border-radius: 4px; }
        .metric { font-weight: bold; color: #2980b9; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #ddd; margin-top: 10px; }
        .narrative { font-style: italic; background-color: #fff3cd; padding: 15px; border-radius: 4px; border: 1px solid #ffeeba; }
        .footer { margin-top: 30px; text-align: center; color: #7f8c8d; font-size: 0.9em; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 0.8em; color: white; background-color: #7f8c8d; }
        .badge-success { background-color: #27ae60; }
        .badge-warning { background-color: #f39c12; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🕵️ Invisible Variables Engine Report</h1>
        <p><strong>Date Generated:</strong> {{ date }}</p>
        
        <div class="card">
            <h3>Automated Executive Summary</h3>
            <p>The engine identified <strong>{{ num_latent_vars }}</strong> potentially "invisible" variable(s) driving systematic model errors. These variables were validated via proxy analysis and model retraining.</p>
        </div>

        <h2>1. Discovered Invisible Variables</h2>
        {% for latent in explanations %}
        <div class="card" style="border-left-color: #e74c3c;">
            <h3>🔍 {{ latent.name }}</h3>
            <div class="narrative">
                {{ latent.narrative }}
            </div>
            
            <h4>Technical Details</h4>
            <ul>
                <li><strong>Validation Status:</strong> {{ latent.validation_note }}</li>
                <li><strong>Proxy Analysis:</strong> {{ latent.proxy_analysis }}</li>
                <li><strong>Affected Context:</strong> {{ latent.context }}</li>
                <li><strong>Impact:</strong> {{ latent.impact }}</li>
            </ul>

            <h4>Visualization: Impact Curve</h4>
            {% if images.get(latent.name + '_impact') %}
            <img src="data:image/png;base64,{{ images.get(latent.name + '_impact') }}" alt="Impact Curve">
            {% else %}
            <p>No impact image available.</p>
            {% endif %}
            
            <h4>Visualization: Distribution</h4>
            {% if images.get(latent.name + '_distribution') %}
            <img src="data:image/png;base64,{{ images.get(latent.name + '_distribution') }}" alt="Distribution">
            {% else %}
            <p>No distribution image available.</p>
            {% endif %}
        </div>
        {% endfor %}

        <h2>2. Global Residual Analysis</h2>
        <p>This heatmap shows where errors are concentrated in the feature space of the original dataset.</p>
        {% if images.get('residual_heatmap') %}
        <img src="data:image/png;base64,{{ images.get('residual_heatmap') }}" alt="Residual Heatmap">
        {% else %}
        <p>No residual heatmap available.</p>
        {% endif %}

        <h2>3. Validation: Retraining Results</h2>
        <p>We retrained the model including the newly discovered latent variables to measure predictive uplift.</p>
        {% if not retraining_df.empty %}
        <table>
            <thead>
                <tr>
                    <th>Baseline RMSE</th>
                    <th>Augmented RMSE</th>
                    <th>Improvement</th>
                    <th>% Improvement</th>
                </tr>
            </thead>
            <tbody>
                {% for index, row in retraining_df.iterrows() %}
                <tr>
                    <td>{{ "%.2f"|format(row['baseline_rmse']) }}</td>
                    <td>{{ "%.2f"|format(row['augmented_rmse']) }}</td>
                    <td>{{ "%.2f"|format(row['rmse_improvement']) }}</td>
                    <td><span class="badge {{ 'badge-success' if row['pct_improvement'] > 1 else 'badge-warning' }}">{{ "%.2f"|format(row['pct_improvement']) }}%</span></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No retraining results available.</p>
        {% endif %}

        <div class="footer">
            Generated by Invisible Variables Engine v1.0
        </div>
    </div>
</body>
</html>
"""

def load_data() -> dict:
    """
    Loads all report data.
    """
    data = {}
    
    # 1. Explanations
    explanations_path = 'data/outputs/latent_variables/explanations.json'
    if os.path.exists(explanations_path):
        with open(explanations_path, 'r') as f:
            data['explanations'] = json.load(f)
    else:
        data['explanations'] = []
        
    # 2. Retraining
    retraining_path = 'data/outputs/latent_variables/retraining_comparison.csv'
    if os.path.exists(retraining_path):
        data['retraining_df'] = pd.read_csv(retraining_path)
    else:
        data['retraining_df'] = pd.DataFrame()
        
    # 3. Images
    data['images'] = {}
    images_dir = 'data/outputs/visualizations'
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.lower().endswith('.png'):
                filepath = os.path.join(images_dir, filename)
                with open(filepath, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                    # Key without extension (e.g., 'residual_heatmap')
                    key = os.path.splitext(filename)[0]
                    data['images'][key] = encoded_string
                    
    return data

def generate_report(data: dict, output_path: str):
    """
    Renders the HTML template.
    """
    template = Template(HTML_TEMPLATE)
    
    html_content = template.render(
        date=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        num_latent_vars=len(data.get('explanations', [])),
        explanations=data.get('explanations', []),
        retraining_df=data.get('retraining_df', pd.DataFrame()),
        images=data.get('images', {})
    )
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        print(f"Successfully generated report at {output_path}")
    except Exception as e:
        raise Exception(f"Error saving report: {e}")

def main():
    """
    Main execution flow.
    """
    output_path = 'data/outputs/reports/IVE_Report.html'
    
    print("-" * 30)
    print("Building Final Report")
    print("-" * 30)
    
    try:
        # 1. Load Data
        data = load_data()
        print(f"Loaded {len(data['explanations'])} explanations.")
        print(f"Loaded {len(data['images'])} images.")
        
        # 2. Generate
        generate_report(data, output_path)
        
        print("\nPipeline completed.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
