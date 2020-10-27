from flask import (Flask, render_template, request)
import configparser
import ipmvp_regression as ipmvp_regression
from helpers import get_api_data
import os

app = Flask("__main__")
config = configparser.ConfigParser()


# Render variable to the html
@app.route("/")
def my_index():
    return render_template("index.html", flask_token="Hello world")

# Get the input form and send the result of the regression
@app.route("/regression", methods = ['GET', 'POST'])
def regression():
    if request.method == "POST":
        box = str(request.json["inputs"]["box"])
        startdate = str(request.json["inputs"]["startdate"])
        weeks = str(request.json["inputs"]["weeks"])

        config.read("inputs_ref.ini")

        config.set("inputs", "box_number", box)
        config.set("inputs", "start", startdate)
        config.set("inputs", "weeks", weeks)

        with open("inputs_ref.ini", 'w') as configfile:
            config.write(configfile)
            configfile.close()

        # # get_api_data(box, startdate, weeks)

        # output_regression = ipmvp_regression.regression_IPMVP('inputs_ref.ini',
        #          'output_ref.ini',
        #          'boudine_kwh.csv')

        # regression_1 = output_regression[0]['regression_1']
        # print(regression_1)
        # regression_2 = output_regression[1]['regression_2']
        # print(regression_2)
        # regression_3 = output_regression[2]['regression_3']
        # print(regression_3)

    regression1 = os.path.join('static', 'regression_1_2019')

    return render_template("index.html", regression1=regression1 )

app.run(debug=True)