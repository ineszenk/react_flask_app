import React, { useState, useInput } from "react";
import logo from "./logo.svg";
import "./App.css";
import axios from "axios";
import useSignUpForm from "./CustomHooks";
import Regression1 from "./assets/figures/regression_1_2019.png";
import Regression2 from "./assets/figures/regression_2_2019.png";
import Regression3 from "./assets/figures/regression_3_2019.png";

function App() {
  const {
    inputs,
    handleInputChange,
    handleSubmit,
    Regression
  } = useSignUpForm();

  console.log(inputs, Regression);

  return (
    <div className="App">
      <header className="App-header">
        <form onSubmit={handleSubmit}>
          <input
            placeholder="box"
            type="text"
            name="box"
            value={inputs.box}
            onChange={handleInputChange}
          />
          <input
            placeholder="start"
            type="text"
            value={inputs.startdate}
            name="startdate"
            onChange={handleInputChange}
          />
          <input
            placeholder="weeks"
            type="integer"
            value={inputs.weeks}
            name="weeks"
            onChange={handleInputChange}
          />
          <button>Submit</button>
        </form>
        <div>
          {Regression ? (
            <div>
              <img src={Regression1} width="300px" height="250px"></img>
              <img src={Regression2} width="300px" height="250px"></img>
              <img src={Regression3} width="300px" height="250px"></img>
            </div>
          ) : (
            ""
          )}
        </div>
      </header>
    </div>
  );
}

export default App;
