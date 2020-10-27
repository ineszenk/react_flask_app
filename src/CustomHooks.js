import React, { useState } from "react";
import axios from "axios";

const useSignUpForm = callback => {
  const [inputs, setInputs] = useState({});
  const [Regression, setRegression] = useState(false);

  const handleSubmit = event => {
    if (event) {
      event.preventDefault();
      setRegression(true);
      //   axios({
      //     method: "post",
      //     url: "http://localhost:5000/regression",
      //     headers: { "content-type": "application/json" },
      //     data: { inputs }
      //   })
      //     .then(setRegression(true))
      //     .catch(error => {
      //       console.log(error);
      //     });
    }
  };
  const handleInputChange = event => {
    event.persist();
    setInputs(inputs => ({
      ...inputs,
      [event.target.name]: event.target.value
    }));
  };
  return {
    handleSubmit,
    handleInputChange,
    inputs,
    Regression
  };
};

export default useSignUpForm;
