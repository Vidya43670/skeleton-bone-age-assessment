import axios from 'axios';

const API_BASE = '/api';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const api = {
  // System Health
  getHealth: async () => {
    const res = await client.get('/health');
    return res.data;
  },

  // Dataset
  getDatasetSummary: async () => {
    const res = await client.get('/dataset/summary');
    return res.data;
  },
  validateDataset: async (payload = {}) => {
    const res = await client.post('/dataset/validate', payload);
    return res.data;
  },
  getDatasetSamples: async (limit = 12) => {
    const res = await client.get(`/dataset/samples?limit=${limit}`);
    return res.data;
  },
  getDatasetSplits: async () => {
    const res = await client.get('/dataset/splits');
    return res.data;
  },

  // Preprocessing
  getPreprocessingMethods: async () => {
    const res = await client.get('/preprocessing/methods');
    return res.data;
  },
  previewPreprocessing: async (formData) => {
    const res = await client.post('/preprocessing/preview', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },

  // Models
  getModels: async () => {
    const res = await client.get('/models');
    return res.data;
  },

  // Experiments
  getExperiments: async () => {
    const res = await client.get('/experiments');
    return res.data;
  },
  getExperiment: async (id) => {
    const res = await client.get(`/experiments/${id}`);
    return res.data;
  },
  trainExperiment: async (id, config = {}) => {
    const res = await client.post(`/experiments/${id}/train`, config);
    return res.data;
  },
  getComparisonResults: async () => {
    const res = await client.get('/results/comparison');
    return res.data;
  },
  getConfusionMatrix: async (id) => {
    const res = await client.get(`/results/confusion-matrix/${id}`);
    return res.data;
  },

  // Prediction Demo
  predict: async (formData) => {
    const res = await client.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return res.data;
  },
};

export default api;
