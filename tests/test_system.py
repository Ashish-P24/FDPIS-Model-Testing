import os
import json
import joblib
import unittest
import sqlite3
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from src.app import app
from src.propagation_engine import FDPISPropagationEngine

class TestFDPISSystem(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_api_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertTrue(data["database_connected"])

    def test_api_overview(self):
        response = self.client.get("/api/overview")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("summary", data)
        self.assertGreater(data["summary"]["total_flights"], 0)
        self.assertGreater(len(data["critical_hubs"]), 0)

    def test_api_search_flights(self):
        response = self.client.get("/api/flights?limit=10")
        self.assertEqual(response.status_code, 200)
        flights = response.json()
        self.assertEqual(len(flights), 10)
        first_flight = flights[0]
        self.assertIn("flight_id", first_flight)
        self.assertIn("delay_prob", first_flight)
        self.assertIn("risk_category", first_flight)

    def test_propagation_engine_graph(self):
        mock_data = pd.DataFrame([
            {
                "flight_id": "TEST_LEG_1",
                "OP_UNIQUE_CARRIER": "DL",
                "ORIGIN": "ATL",
                "DEST": "LGA",
                "TAIL_NUM": "N123AA",
                "CRS_DEP_TIME": 800,
                "CRS_ARR_TIME": 1000,
                "DISTANCE": 760,
                "predicted_delay": 60.0,
                "delay_prob": 0.75
            },
            {
                "flight_id": "TEST_LEG_2",
                "OP_UNIQUE_CARRIER": "DL",
                "ORIGIN": "LGA",
                "DEST": "BOS",
                "TAIL_NUM": "N123AA",
                "CRS_DEP_TIME": 1045,
                "CRS_ARR_TIME": 1200,
                "DISTANCE": 214,
                "predicted_delay": 0.0,
                "delay_prob": 0.20
            }
        ])
        
        engine = FDPISPropagationEngine(turnaround_buffer_threshold=45)
        n_nodes, n_edges = engine.build_network_graph(mock_data)
        self.assertEqual(n_nodes, 2)
        self.assertEqual(n_edges, 1)
        
        tree = engine.propagate_cascade("TEST_LEG_1", initial_delay_minutes=60.0)
        self.assertIsNotNone(tree)
        self.assertEqual(tree["affected_flights_count"], 1)
        self.assertEqual(tree["edges"][0]["slack_minutes"], 0)
        self.assertGreater(tree["nodes"][1]["residual_delay"], 0)

    def test_model_artifacts_loaded(self):
        self.assertTrue(os.path.exists("models/lightgbm.pkl"))
        self.assertTrue(os.path.exists("models/delay_regressor.pkl"))
        self.assertTrue(os.path.exists("models/features.json"))
        self.assertTrue(os.path.exists("results/final_test_evaluation.json"))
        self.assertTrue(os.path.exists("results/final_test_regression.json"))

if __name__ == "__main__":
    unittest.main()
