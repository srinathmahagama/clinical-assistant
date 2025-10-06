import React, { useState } from 'react';
import { integrateServices } from '../services/integrationService';

const IntegrationPage: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const data = await integrateServices(inputText);
      setResult(data);
      console.log('Integration result:', data);
    } catch (err) {
      console.error(err);
      alert('Error processing your input.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Enter text or symptoms"
      />
      <button onClick={handleSubmit} disabled={loading}>
        Submit
      </button>

      {result && (
        <pre>{JSON.stringify(result, null, 2)}</pre>
      )}
    </div>
  );
};

export default IntegrationPage;
