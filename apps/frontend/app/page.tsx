"use client";

import Link from "next/link";
import { FC, useEffect, useState } from "react";

const Home: FC = () => {

  const [data, setData] = useState<any>(null);
  const [nlpData, setnlpData] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchData();
  }, [data])

  async function fetchData() {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/test-backend");
      const nlpRes = await fetch("http://localhost:8000/test-nlp-service");
      const json = await res.json();
      const nlpJson = await nlpRes.json();
      setData(json);
      setnlpData(nlpJson)
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto p-8">
      <h1 className="text-4xl font-bold mb-4">Welcome to Clinical Assistant</h1>
      <p className="mb-6 text-lg">
        This app helps Indigenous users get symptom-based recommendations.
      </p>
      <p className="mb-6 text-lg text-red-300">
          {data}
      </p> <p className="mb-6 text-lg text-blue-300">
          {nlpData}
      </p>
      
      <nav className="space-x-4 text-blue-600">
        <Link href="./about" className="hover:underline">About</Link>
      </nav>
    </div>
  );
};

export default Home;
