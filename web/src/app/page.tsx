'use client';
import { useState } from 'react';

export default function Home() {
  const placeholders: Record<string, number> = {
    LotArea: 8450,
    OverallQual: 7,
    YearBuilt: 2003,
    TotalBsmtSF: 856,
    GrLivArea: 1710,
    FullBath: 2,
    BedroomAbvGr: 3,
    GarageCars: 2,
  };

  const [formData, setFormData] = useState<Record<string, string>>({
    LotArea: '',
    OverallQual: '',
    YearBuilt: '',
    TotalBsmtSF: '',
    GrLivArea: '',
    FullBath: '',
    BedroomAbvGr: '',
    GarageCars: '',
  });

  const [price, setPrice] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const estimate = async () => {
    setLoading(true);
    // Use the value if typed, otherwise fall back to the placeholder value
    const dataToSend = Object.keys(placeholders).reduce((acc, key) => {
      acc[key] = formData[key] === '' ? placeholders[key] : parseFloat(formData[key]);
      return acc;
    }, {} as Record<string, number>);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(dataToSend),
      });
      const data = await response.json();
      setPrice(data.predicted_price);
    } catch (error) {
      console.error("Failed to fetch prediction", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="max-w-3xl w-full bg-white shadow-xl rounded-2xl p-8 border border-gray-100">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-extrabold text-gray-900 mb-2">Real Estate AI</h1>
          <p className="text-gray-500">Automated Property Valuation powered by Random Forest</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { label: 'Lot Area (sq ft)', name: 'LotArea', min: 1000 },
            { label: 'Overall Quality (1-10)', name: 'OverallQual', min: 1, max: 10 },
            { label: 'Year Built', name: 'YearBuilt', min: 1800, max: 2025 },
            { label: 'Total Basement (sq ft)', name: 'TotalBsmtSF', min: 0 },
            { label: 'Living Area (sq ft)', name: 'GrLivArea', min: 500 },
            { label: 'Full Bathrooms', name: 'FullBath', min: 1 },
            { label: 'Bedrooms', name: 'BedroomAbvGr', min: 1 },
            { label: 'Garage Capacity (Cars)', name: 'GarageCars', min: 0 },
          ].map((field) => (
            <div key={field.name} className="flex flex-col">
              <label className="text-sm font-semibold text-gray-700 mb-2">{field.label}</label>
              <input
                type="number"
                name={field.name}
                value={formData[field.name]}
                placeholder={placeholders[field.name].toString()}
                onChange={handleChange}
                min={field.min}
                max={field.max}
                className="p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition text-black placeholder:text-[#bebebe]"
              />
            </div>
          ))}
        </div>

        <div className="mt-8 text-center">
          <button
            onClick={estimate}
            disabled={loading}
            className="w-full md:w-1/2 py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-xl shadow-lg transition-all transform hover:scale-105"
          >
            {loading ? 'Analyzing Market Data...' : 'Estimate Property Value'}
          </button>
        </div>

        {price && (
          <div className="mt-8 p-6 bg-green-50 border border-green-200 rounded-xl text-center animate-fade-in-up">
            <h2 className="text-sm font-semibold text-green-600 uppercase tracking-wide">Estimated Market Value</h2>
            <div className="text-5xl font-extrabold text-green-800 mt-2">
              ₹{price.toLocaleString('en-IN')}
            </div>
            <p className="text-xs text-green-600 opacity-80 mt-3">
              Based on historical data and current model calibration.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
