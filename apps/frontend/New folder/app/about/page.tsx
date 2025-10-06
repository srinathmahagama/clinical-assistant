import Link from "next/link";
import { FC } from "react";

const About: FC = () => {
  return (
    <div className="max-w-3xl mx-auto p-8">
      <h1 className="text-4xl font-bold mb-4">About Clinical Assistant</h1>
      <p className="mb-6 text-lg">
        This app is designed to assist Indigenous communities with symptom
        tracking and recommendations using AI-powered NLP and ML models.
      </p>
      <h1 className="text-4xl font-bold mb-4">How it Works</h1>
      <ol className="list-decimal list-inside mb-6 space-y-2 text-lg">
        <li>User inputs symptoms via text or audio.</li>
        <li>AI translates input (if necessary) to English.</li>
        <li>Symptoms are extracted and disease is predicted.</li>
        <li>Recommendations are provided based on the prediction.</li>
      </ol>
      <Link href="/" className="text-blue-600 hover:underline">
        Back to Home
      </Link>
    </div>
  );
};

export default About;
