export interface IntegrationResponse {
    nlp_result: any;
    ml_result: any;
  }
  
  export const integrateServices = async (text: string, language = 'noongar'): Promise<IntegrationResponse> => {
    const response = await fetch('http://localhost:8000/integrate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, language }),
    });
  
    if (!response.ok) {
      throw new Error('Integration API failed');
    }
  
    return response.json();
  };
  