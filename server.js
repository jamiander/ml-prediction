import express from 'express';
import cors from 'cors';
import fs from 'fs';
import { createRequire } from 'module';
import GraduationPredictor from './models/GraduationPredictor.js';

// Create require for csv-parser which doesn't support ES modules
const require = createRequire(import.meta.url);
const csvParser = require('csv-parser');

const app = express();
const port = 5000;

app.use(cors());

let studentData = [];
const predictor = new GraduationPredictor();

fs.createReadStream('data/student-data')
    .pipe(csvParser({
        separator: ' ',
        headers: ['grade', 'employed', 'age', 'married', 'graduate']
    }))
    .on('data', (row) => {
        studentData.push({
            grade: row.grade,
            employed: row.employed === 'true' ? 1 : 0,
            age: row.age || '25',  // Default age if not provided
            married: row.married === 'true' ? 1 : 0,
            graduate: row.graduate === 'true' ? 1 : 0
        });
    })
    .on('end', () => {
        predictor.train(studentData);
        console.log('Data loaded and model trained');
    });

app.get('/data', (req, res) => {
    res.json({
        rawData: studentData,
        modelTrained: predictor.model !== null,
        dataPoints: studentData.length
    });
});

app.get('/test', (req, res) => {
    if (!predictor.model) {
        return res.json({
            status: 'error',
            message: 'Model not trained',
            dataPoints: studentData.length
        });
    }

    try {
        // Test predictions on a range of grades with both employed states
        const testGrades = [1, 2, 3, 4];
        const testAges = [20, 25, 30, 40];
        const results = testGrades.flatMap(grade => 
            testAges.flatMap(age => 
                [0, 1].flatMap(employed => 
                    [0, 1].map(married => {
                        const prediction = predictor.predict(grade, employed, age, married);
                        return {
                            grade,
                            age,
                            employed,
                            married,
                            graduationProbability: prediction ? prediction.probability : null,
                            willGraduate: prediction ? prediction.willGraduate : null
                        };
                    })
                )
            )
        );

        res.json({
            status: 'success',
            modelTrained: true,
            testResults: results,
            modelStats: predictor.stats,
            dataPoints: studentData.length,
            sampleData: studentData.slice(0, 3)
        });
    } catch (error) {
        res.json({
            status: 'error',
            message: 'Error testing model',
            error: error.message
        });
    }
});

app.get('/predict', (req, res) => {
    console.log(req.query);
    const grade = req.query.grade;
    const employed = req.query.employed === '1' || req.query.employed === 'true' ? 1 : 0;
    const age = req.query.age || 25;
    const married = req.query.married === '1' || req.query.married === 'true' ? 1 : 0;
    
    const prediction = predictor.predict(grade, employed, age, married);
    
    if (!prediction) {
        return res.json({
            status: 'error',
            message: 'Model not trained or invalid prediction',
            modelType: 'not-trained',
            dataPoints: studentData.length
        });
    }
    
    res.json({
        status: 'success',
        grade: Number(grade),
        employed,
        age: Number(age),
        married,
        graduationProbability: prediction.probability,
        willGraduate: prediction.willGraduate,
        modelType: 'polynomial-regression',
        dataPoints: studentData.length
    });
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});




