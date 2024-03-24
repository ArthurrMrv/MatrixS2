function createMatrix(parent, m = 2, n = 2, values = [[]]) {

    let id = Date.now();

    let container = parent.appendChild(document.createElement("table"));
    container.classList.add("matrix-container");
    container.setAttribute("id-matrix", id);
    container.setAttribute("nbRows", m);
    container.setAttribute("nbCols", n);

    let matrixAndChangeRows = container.appendChild(document.createElement("tr"));
    matrixAndChangeRows.classList.add("matrix-and-change-rows");

    if (matrices.length >= 1) {
        let tdOperant = matrixAndChangeRows.appendChild(document.createElement("td"));
        let options = [" ", "+", "-", "*"]
        let operandMatrix = tdOperant.appendChild(document.createElement("select"));
        for (let i = 0; i < options.length; i++) {
            let option = operandMatrix.appendChild(document.createElement("option"));
            option.setAttribute("value", options[i]);
            option.innerText = options[i];
        }
        operandMatrix.classList.add("operand-matrix");
    }

    let tdMatrix = matrixAndChangeRows.appendChild(document.createElement("td"));
    tdMatrix.classList.add("td-matrix");

    let tdChangeRows = matrixAndChangeRows.appendChild(document.createElement("td"));
    tdChangeRows.classList.add("td-change-rows");

    let tdChangeColumns = container.appendChild(document.createElement("tr"));
    tdChangeColumns.classList.add("td-change-columns");

    let matrix = tdMatrix.appendChild(document.createElement("div"));
    matrix.classList.add("matrix");

    let changeRows = tdChangeRows.appendChild(document.createElement("div"));
    changeRows.classList.add("change-rows");

    let changeColumns = tdChangeColumns.appendChild(document.createElement("div"));
    changeColumns.classList.add("change-columns");

    for (let i = 0; i < m; i++) {
        let row = document.createElement("div");
        row.classList.add("row");
        for (let j = 0; j < n; j++) {
            row.appendChild(createCell(i, j, values.length == m && values[0].length == n ? values[i][j] : 0));
        }
        matrix.appendChild(row);
    }

    let addRowButton = changeRows.appendChild(document.createElement("button"));
    addRowButton.classList.add("add-row");
    addRowButton.setAttribute("id-matrix", `${id}`);
    addRowButton.addEventListener("click", addRow);
    addRowButton.innerText = "+";

    let deleteRowButton = changeRows.appendChild(document.createElement("button"));
    deleteRowButton.classList.add("delete-row");
    deleteRowButton.setAttribute("id-matrix", `${id}`);
    deleteRowButton.addEventListener("click", delRow);
    deleteRowButton.innerText = "-";

    let addColumnButton = changeColumns.appendChild(document.createElement("button"));
    addColumnButton.classList.add("add-column");
    addColumnButton.setAttribute("id-matrix", `${id}`);
    addColumnButton.addEventListener("click", addCol);
    addColumnButton.innerText = "+";

    let deleteColumnButton = changeColumns.appendChild(document.createElement("button"));
    deleteColumnButton.classList.add("delete-column");
    deleteColumnButton.setAttribute("id-matrix", `${id}`);
    deleteColumnButton.addEventListener("click", delCol);
    deleteColumnButton.innerText = "-";

    if (matrices.length >= 1) {
        let tdrmMatrix = matrixAndChangeRows.appendChild(document.createElement("td"));
        let removeMatrixButton = tdrmMatrix.appendChild(document.createElement("button"));
        removeMatrixButton.textContent = "\u{1f5d1}";
        removeMatrixButton.addEventListener("click", function () { return removeMatrix(id) });
        removeMatrixButton.classList.add("remove-matrix");
        removeMatrixButton.classList.remove("hidden");
    }

    matrices.push(id);

    return container;
}

function matrixAnswer(parent, m = 2, n = 2, values = [[]]) {

    let id = Date.now();

    let container = parent.appendChild(document.createElement("table"));
    container.classList.add("matrix-container");
    container.setAttribute("id-matrix", id);
    container.setAttribute("nbRows", m);
    container.setAttribute("nbCols", n);

    let matrixAndChangeRows = container.appendChild(document.createElement("tr"));
    matrixAndChangeRows.classList.add("matrix-and-change-rows");

    let tdMatrix = matrixAndChangeRows.appendChild(document.createElement("td"));
    tdMatrix.classList.add("td-matrix");

    let tdChangeRows = matrixAndChangeRows.appendChild(document.createElement("td"));
    tdChangeRows.classList.add("td-change-rows");

    let tdChangeColumns = container.appendChild(document.createElement("tr"));
    tdChangeColumns.classList.add("td-change-columns");

    let matrix = tdMatrix.appendChild(document.createElement("div"));
    matrix.classList.add("matrix");

    let changeRows = tdChangeRows.appendChild(document.createElement("div"));
    changeRows.classList.add("change-rows");

    let changeColumns = tdChangeColumns.appendChild(document.createElement("div"));
    changeColumns.classList.add("change-columns");

    for (let i = 0; i < m; i++) {
        let row = document.createElement("div");
        row.classList.add("row");
        for (let j = 0; j < n; j++) {
            row.appendChild(createCell(i, j, values.length == m && values[0].length == n ? values[i][j] : 0));
        }
        matrix.appendChild(row);
    }

    return container;
}

function addCol() {
    const idMatrix = this.getAttribute("id-matrix");
    let matrix = document.querySelector(`table[id-matrix="${idMatrix}"]`);

    const m = matrix.getAttribute("nbRows");
    let rows = matrix.querySelectorAll(".row");

    for (let i = 0; i < m; i++) {
        rows[i].appendChild(createCell(i, rows[i].children.length)); // Get the current number of columns in this row
    }

    matrix.setAttribute("nbCols", parseInt(matrix.getAttribute("nbCols")) + 1); // Increment the number of columns
}


function delCol() {
    const idMatrix = this.getAttribute("id-matrix");
    let matrix = document.querySelector(`table[id-matrix="${idMatrix}"]`);

    const m = matrix.getAttribute("nbRows");

    if (matrix.getAttribute("nbCols") < 2) {
        return undefined;
    }

    let rows = matrix.querySelectorAll(".row");
    // remove the last element of rows 
    for (let i = 0; i < m; i++) {
        rows[i].removeChild(rows[i].lastChild);
    }

    matrix.setAttribute("nbCols", matrix.getAttribute("nbCols") - 1);
}

function addRow() {
    const idMatrix = this.getAttribute("id-matrix");
    let matrix = document.querySelector(`table[id-matrix="${idMatrix}"]`);

    const n = matrix.getAttribute("nbCols");

    let row = document.createElement("div");
    row.classList.add("row");

    for (let j = 0; j < n; j++) {
        row.appendChild(createCell(matrix.getAttribute("nbRows"), j)); // Use the current number of rows
    }
    matrix.querySelector('.matrix').appendChild(row); // Append the row to the matrix container

    matrix.setAttribute("nbRows", parseInt(matrix.getAttribute("nbRows")) + 1); // Increment the number of rows
}


function delRow() {
    const idMatrix = this.getAttribute("id-matrix");
    let matrix = document.querySelector(`table[id-matrix="${idMatrix}"]`);

    const n = matrix.getAttribute("nbCols");

    if (matrix.getAttribute("nbRows") < 2) {
        return undefined;
    }

    let rows = matrix.querySelectorAll('.row');
    matrix.querySelector('.matrix').removeChild(rows[rows.length - 1]); // Remove the last row

    matrix.setAttribute("nbRows", parseInt(matrix.getAttribute("nbRows")) - 1); // Decrement the number of rows
}

function createCell(i, j, value = 0) {
    let input = document.createElement("input");
    input.setAttribute("type", "text");
    input.setAttribute("value", `${value}`);
    input.setAttribute("value-row", i);
    input.setAttribute("value-col", j);
    input.classList.add("cell")
    return input
}

function createButtons(buttonsList) {
    let parent = document.querySelector("div#buttons");

    for (let button in buttonsList) {
        let newButton = document.createElement("button");
        newButton.innerText = buttonsList[button].text;
        newButton.setAttribute("id", "function-button");
        newButton.setAttribute("nb-matrices", buttonsList[button].nbMatrices);
        newButton.addEventListener("click", function () { return apiCall(buttonsList[button].functionID) });
        parent.appendChild(newButton);
    }

    let newButton = document.createElement("button");
    newButton.innerText = "Clear";
    newButton.setAttribute("id", "function-button");
    newButton.setAttribute("nb-matrices", undefined);
    newButton.addEventListener("click", doClear);
    parent.appendChild(newButton);
}

function setButtons() {
    // let addMatrixButton = document.querySelector("#add-matrix");
    // addMatrixButton.addEventListener("click", addMatrix);

    // let buttons = document.querySelector("div#buttons").innerHTML;

    // get all the childs of the div#buttons
    let buttons = document.querySelectorAll("#function-button")

    for (let i = 0; i < buttons.length; i++) {
        console.log(buttons[i].getAttribute("nb-matrices"), matrices.length, buttons[i].getAttribute("nb-matrices") == matrices.length);
        if (buttons[i].getAttribute("nb-matrices") == matrices.length || buttons[i].getAttribute("nb-matrices") == "undefined") {

            if (!buttons[i].classList.contains("enabled")) {
                buttons[i].classList.add("enabled");
            }
        } else {
            if (buttons[i].classList.contains("enabled")) {
                buttons[i].classList.remove("enabled");
            }
        }

    }

    let addMatrixButton = document.querySelector("div#calculus").querySelector("#addMatrix-button");
    if (addMatrixButton != null) {
        document.querySelector("div#calculus").removeChild(addMatrixButton);
        addMatrixButton = null;
    }
    if (matrices.length < NB_MAX_MATRICES) {
        if (addMatrixButton == null) {
            createAddMatrixButton();
        }
    }

}


function matrixToJson(idMatrix) {
    let matrix = document.querySelector(`table[id-matrix="${idMatrix}"]`);
    let rows = matrix.querySelectorAll(".row");
    let m = rows.length;
    let n = rows[0].children.length;
    let matrixJson = [];

    for (let i = 0; i < m; i++) {
        let row = [];
        for (let j = 0; j < n; j++) {
            row.push(parseFloat(rows[i].children[j].value));
        }
        matrixJson.push(row);
    }

    return matrixJson;
}


function apiCall(method) {
    let data_to_send = {};
    data_to_send["matrices"] = dataMatrices();
    data_to_send["method"] = method;

    // Convert JavaScript object to JSON string
    let json_data = JSON.stringify(data_to_send);

    // Send AJAX request to Flask app
    $.ajax({
        type: 'POST',
        url: '/call_function', // Change this to your Flask route
        contentType: 'application/json',
        data: json_data,
        success: function (response) {
            console.log("Data sent successfully");

            // Handle response from Flask app if needed
            parent = document.querySelector("p#ans");
            parent.innerHTML = "";
            console.log(response);
            if (response["type"] == "matrix") {
                matrixAnswer(parent, response["content"].length, response["content"][0].length, response["content"]);
                console.log(response);
            } else if (response["type"] == "number") {
                let p = parent.appendChild(document.createElement("p"));
                p.innerText = response["content"];
            } else if (response["type"] == "list[number]") {
                for (let i = 0; i < response["content"].length; i++) {
                    let p = parent.appendChild(document.createElement("p"));
                    p.innerText = response["content"][i];
                }
            } else if (response["type"] == "text") {
                let p = parent.appendChild(document.createElement("p"));
                p.innerText = response["content"];
            } else if (response["type"] == "list[matrix]") {
                for (let i = 0; i < response["content"].length; i++) {
                    matrixAnswer(parent, response["content"][i].length, response["content"][i][0].length, response["content"][i]);
                }
            } else {
                console.log("Unknown type of response");
            }

        },
        error: function (error) {
            console.error("Error sending data: ", error);
        }
    })
}

function dataMatrices() {
    let data = {};

    for (let n = 0; n < matrices.length; n++) {
        // set all values to 0
        let matrix = document.querySelector(`table[id-matrix="${matrices[n]}"]`);
        let rows = matrix.querySelectorAll(".row");
        let m = rows.length;
        let nb_rows = rows[0].children.length;

        data[`${matrices[n]}`] = [];

        for (let i = 0; i < m; i++) {
            let row_values = [];
            for (let j = 0; j < nb_rows; j++) {
                row_values.push(rows[i].children[j].value);
            }
            data[`${matrices[n]}`].push(row_values);
        }
    }


    return data;
}

function doClear() {
    for (let i = 1; i < matrices.length; i++) {
        removeMatrix(matrices[i]);
    }

    // set all values to 0
    let matrix = document.querySelector(`table[id-matrix="${matrices[0]}"]`);
    let rows = matrix.querySelectorAll(".row");
    let m = rows.length;
    let n = rows[0].children.length;

    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            rows[i].children[j].value = 0;
        }
    }

    let parent = document.querySelector("div#ans-query");
    parent.childNodes.forEach((child) => {
        parent.removeChild(child);
    });
}

function addMatrix() {
    let id = matrices.length;
    let m = 2;
    let n = 2;
    createMatrix(MatricesParent, id, m, n);

    setButtons();
    // 
}

function removeMatrix(idMatrix) {

    console.log("removing matrix", idMatrix);
    console.log(matrices);
    matrices = matrices.filter((matrix) => matrix != idMatrix);
    let matrixParent = document.querySelector(`table[id-matrix="${idMatrix}"]`).parentNode;

    matrixParent.removeChild(document.querySelector(`table[id-matrix="${idMatrix}"]`));

    setButtons();
}

function createAddMatrixButton() {
    let addMatrixButton = document.querySelector("div#calculus").appendChild(document.createElement("button"));
    addMatrixButton.innerText = "+";
    addMatrixButton.addEventListener("click", addMatrix);
    addMatrixButton.setAttribute("id", "addMatrix-button");
    return addMatrixButton;
}

let buttonsToCreate = {
    add: {
        functionID: "add",
        text: "Add",
        nbMatrices: 2
    },
    subtract: {
        functionID: "substract",
        text: "Subtract",
        nbMatrices: 2
    },
    multiply: {
        functionID: "mulptiply",
        text: "Multiply",
        nbMatrices: 2
    },
    transpose: {
        functionID: "transpose",
        text: "Transpose",
        nbMatrices: 1
    },
    determinant: {
        functionID: "determinant",
        text: "Determinant",
        nbMatrices: 1
    },
    inverse: {
        functionID: "inverse",
        text: "Inverse",
        nbMatrices: 1
    },
    eigenvalues: {
        functionID: "eigenvalues",
        text: "Eigenvalues",
        nbMatrices: 1
    },
    eigenvectors: {
        functionID: "eigenvectors",
        text: "Eigenvectors",
        nbMatrices: 1
    },
}

let NB_MAX_MATRICES = 2;

let matrices = [];
MatricesParent = document.querySelector("div#calculus");

// Create initial matrix
createButtons(buttonsToCreate)

createMatrix(MatricesParent);

// Add a button to add matrices

createAddMatrixButton();

setButtons()