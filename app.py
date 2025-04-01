from flask import Flask, render_template, send_from_directory

from data_processing import eigenvalues, eigenvectors, mse_scores, cluster_ids

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scree')
def scree():
    return send_from_directory('static', 'scree.csv')

@app.route('/mse')
def mse():
    return send_from_directory('static', 'mse_scores.csv')

@app.route('/biplot')
def biplot():
    return send_from_directory('static', 'biplot.csv')

@app.route('/loadings')
def loadings():
    return send_from_directory('static', 'loadings.csv')

@app.route('/data')
def data():
    return send_from_directory('static', 'spotify-clustered.csv')

@app.route('/mds-data')
def mds_data():
    return send_from_directory('static', 'data_mds.csv')

@app.route('/mds-var')
def mds_var():
    return send_from_directory('static', 'var_mds.csv')

if __name__ == '__main__':
    app.run(debug=True, port=8000)