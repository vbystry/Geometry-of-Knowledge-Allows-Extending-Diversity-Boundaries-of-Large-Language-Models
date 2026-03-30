#!/usr/bin/env python3
"""
Plot lambda ablation study results comparing mean_distinct and mean_utility
across different lambda values.
Creates an HTML file with interactive charts using Chart.js (no dependencies needed).
"""

import json
from pathlib import Path

# Data from the ablation study - updated with all lambda values
data = {
    0.5: {"mean_distinct": 6.33, "mean_utility": 4.272856003226969},
    1.5: {"mean_distinct": 7.88, "mean_utility": 4.540860849637965},
    2.0: {"mean_distinct": 8.48, "mean_utility": 4.613310582770313},
    3.0: {"mean_distinct": 8.91, "mean_utility": 4.702623492574485},
    4.0: {"mean_distinct": 9.11, "mean_utility": 4.7279042639111575},
    5.0: {"mean_distinct": 9.35, "mean_utility": 4.712891712155291},
    6.0: {"mean_distinct": 9.44, "mean_utility": 4.783117301517427},
    7.0: {"mean_distinct": 9.53, "mean_utility": 4.761486114410932},
    8.0: {"mean_distinct": 9.68, "mean_utility": 4.773613563941272},
    9.0: {"mean_distinct": 9.55, "mean_utility": 4.736098396227802},
    10.0: {"mean_distinct": 9.76, "mean_utility": 4.7839000159356315},
}

# Extract data for plotting
lambdas = sorted(data.keys())
distinct = [data[lam]["mean_distinct"] for lam in lambdas]
utility = [data[lam]["mean_utility"] for lam in lambdas]

# Find optimal values
max_distinct_idx = distinct.index(max(distinct))
max_utility_idx = utility.index(max(utility))
max_distinct_lambda = lambdas[max_distinct_idx]
max_utility_lambda = lambdas[max_utility_idx]

# Calculate percentage improvements
utility_improvement = ((max(utility) - min(utility)) / min(utility) * 100)
distinct_improvement = ((max(distinct) - min(distinct)) / min(distinct) * 100)

# Create HTML with Chart.js
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Lambda Ablation Study</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #fff9c4;
            font-weight: bold;
        }}
        .summary-box {{
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Lambda Ablation Study: Distinct vs Utility</h1>
    
    <div class="container">
        <div class="summary-box">
            <h3>Key Findings</h3>
            <ul>
                <li><strong>Best Utility:</strong> λ = {max_utility_lambda} (Utility: {data[max_utility_lambda]['mean_utility']:.4f})</li>
                <li><strong>Best Distinct:</strong> λ = {max_distinct_lambda} (Distinct: {data[max_distinct_lambda]['mean_distinct']:.2f})</li>
                <li><strong>Utility Range:</strong> {min(utility):.4f} - {max(utility):.4f} ({utility_improvement:.1f}% improvement)</li>
                <li><strong>Distinct Range:</strong> {min(distinct):.2f} - {max(distinct):.2f} ({distinct_improvement:.1f}% improvement)</li>
            </ul>
        </div>
    </div>
    
    <div class="container">
        <h2>Mean Distinct vs Lambda</h2>
        <div class="chart-container">
            <canvas id="distinctChart"></canvas>
        </div>
    </div>
    
    <div class="container">
        <h2>Mean Utility vs Lambda</h2>
        <div class="chart-container">
            <canvas id="utilityChart"></canvas>
        </div>
    </div>
    
    <div class="container">
        <h2>Combined View</h2>
        <div class="chart-container">
            <canvas id="combinedChart"></canvas>
        </div>
    </div>
    
    <div class="container">
        <h2>Summary Statistics</h2>
        <table>
            <thead>
                <tr>
                    <th>Lambda (λ)</th>
                    <th>Mean Distinct</th>
                    <th>Mean Utility</th>
                </tr>
            </thead>
            <tbody>
                {''.join(['<tr' + (' class="highlight"' if lam == max_utility_lambda or lam == max_distinct_lambda else '') + f'><td>{lam}</td><td>{data[lam]["mean_distinct"]:.2f}</td><td>{data[lam]["mean_utility"]:.4f}</td></tr>' for lam in lambdas])}
            </tbody>
        </table>
    </div>

    <script>
        const lambdas = {json.dumps(lambdas)};
        const distinct = {json.dumps(distinct)};
        const utility = {json.dumps(utility)};
        
        // Chart 1: Mean Distinct
        const distinctCtx = document.getElementById('distinctChart').getContext('2d');
        new Chart(distinctCtx, {{
            type: 'line',
            data: {{
                labels: lambdas.map(l => l.toString()),
                datasets: [{{
                    label: 'Mean Distinct',
                    data: distinct,
                    borderColor: '#2E86AB',
                    backgroundColor: 'rgba(46, 134, 171, 0.1)',
                    borderWidth: 3,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Mean Distinct vs Lambda',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Lambda (λ)',
                            font: {{ size: 12, weight: 'bold' }}
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Mean Distinct',
                            font: {{ size: 12, weight: 'bold' }}
                        }},
                        beginAtZero: false
                    }}
                }}
            }}
        }});
        
        // Chart 2: Mean Utility
        const utilityCtx = document.getElementById('utilityChart').getContext('2d');
        new Chart(utilityCtx, {{
            type: 'line',
            data: {{
                labels: lambdas.map(l => l.toString()),
                datasets: [{{
                    label: 'Mean Utility',
                    data: utility,
                    borderColor: '#A23B72',
                    backgroundColor: 'rgba(162, 59, 114, 0.1)',
                    borderWidth: 3,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Mean Utility vs Lambda',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Lambda (λ)',
                            font: {{ size: 12, weight: 'bold' }}
                        }}
                    }},
                    y: {{
                        title: {{
                            display: true,
                            text: 'Mean Utility',
                            font: {{ size: 12, weight: 'bold' }}
                        }},
                        beginAtZero: false
                    }}
                }}
            }}
        }});
        
        // Chart 3: Combined
        const combinedCtx = document.getElementById('combinedChart').getContext('2d');
        new Chart(combinedCtx, {{
            type: 'line',
            data: {{
                labels: lambdas.map(l => l.toString()),
                datasets: [
                    {{
                        label: 'Mean Distinct',
                        data: distinct,
                        borderColor: '#2E86AB',
                        backgroundColor: 'rgba(46, 134, 171, 0.1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        yAxisID: 'y',
                        tension: 0.3
                    }},
                    {{
                        label: 'Mean Utility',
                        data: utility,
                        borderColor: '#A23B72',
                        backgroundColor: 'rgba(162, 59, 114, 0.1)',
                        borderWidth: 3,
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        yAxisID: 'y1',
                        tension: 0.3
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Lambda Ablation Study: Distinct vs Utility',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: true
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Lambda (λ)',
                            font: {{ size: 12, weight: 'bold' }}
                        }}
                    }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {{
                            display: true,
                            text: 'Mean Distinct',
                            font: {{ size: 12, weight: 'bold' }},
                            color: '#2E86AB'
                        }},
                        ticks: {{
                            color: '#2E86AB'
                        }},
                        beginAtZero: false
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {{
                            display: true,
                            text: 'Mean Utility',
                            font: {{ size: 12, weight: 'bold' }},
                            color: '#A23B72'
                        }},
                        ticks: {{
                            color: '#A23B72'
                        }},
                        grid: {{
                            drawOnChartArea: false,
                        }},
                        beginAtZero: false
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

# Write HTML file
with open('lambda_ablation_chart.html', 'w') as f:
    f.write(html_content)

print("HTML chart saved as 'lambda_ablation_chart.html'")
print("Open it in a web browser to view the interactive charts.")

# Print summary statistics
print("\n" + "="*70)
print("Lambda Ablation Study Summary (Complete)")
print("="*70)
print(f"{'Lambda':<10} {'Mean Distinct':<15} {'Mean Utility':<15}")
print("-"*70)
for lam in lambdas:
    marker = ""
    if lam == max_utility_lambda:
        marker = " ← Best Utility"
    elif lam == max_distinct_lambda:
        marker = " ← Best Distinct"
    print(f"{lam:<10.1f} {data[lam]['mean_distinct']:<15.2f} {data[lam]['mean_utility']:<15.4f}{marker}")
print("="*70)
print(f"\nBest Utility: λ = {max_utility_lambda} (Utility: {data[max_utility_lambda]['mean_utility']:.4f})")
print(f"Best Distinct: λ = {max_distinct_lambda} (Distinct: {data[max_distinct_lambda]['mean_distinct']:.2f})")
