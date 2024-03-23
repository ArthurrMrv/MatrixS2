import { spawn as spawner } from 'child_process';

const data_to_pass_in = {message: 'data to pass in'};

console.log('Data sent to script: ', data_to_pass_in);

const pythonProcess = spawner('python', ['web/script.py', JSON.stringify(data_to_pass_in)]);


pythonProcess.stdout.on('data', (data) => {
    let receivedData = ''; // Variable to store received data

    receivedData = JSON.parse(data.toString());
    console.log('Data from script: ', receivedData);



    console.log('Data received from script: ', receivedData['1']);
});

