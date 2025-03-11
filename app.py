from flask import Flask, render_template, send_from_directory

from data_processing import eigenvalues, eigenvectors, mse_scores, cluster_ids

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scree')
def scree():
    return send_from_directory('static', 'scree.csv')

if __name__ == '__main__':
    app.run(debug=True, port=8000)