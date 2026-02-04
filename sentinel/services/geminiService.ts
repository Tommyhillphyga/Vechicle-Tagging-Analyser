
import { GoogleGenAI, Type } from "@google/genai";
import { MatchResult, Snapshot } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const ANALYSIS_PROMPT = `
Analyze these two sets of images: "Entry" and "Exit".
Your task is to act as a high-precision computer vision pipeline.
1. Identify the vehicles in both sets (Make, Model, Color, Plate).
2. Identify the driver's face in each vehicle.
3. Compare the driver in the Entry image with the driver in the Exit image for the SAME vehicle.
4. If the driver faces do not match, flag as a MISMATCH.
5. Provide a similarity score (0.0 to 1.0) for both the vehicle and the driver.

Return a JSON array of MatchResults.
`;

export const analyzeTraffic = async (entryImages: string[], exitImages: string[]): Promise<MatchResult[]> => {
  try {
    const entryParts = entryImages.map(img => ({
      inlineData: { mimeType: 'image/jpeg', data: img.split(',')[1] }
    }));
    const exitParts = exitImages.map(img => ({
      inlineData: { mimeType: 'image/jpeg', data: img.split(',')[1] }
    }));

    // Use gemini-3-pro-preview for complex multi-modal reasoning and forensic comparison tasks
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: {
        parts: [
          { text: ANALYSIS_PROMPT },
          { text: "Entry Images:" },
          ...entryParts,
          { text: "Exit Images:" },
          ...exitParts,
        ]
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.STRING },
              entrySnapshotId: { type: Type.STRING },
              exitSnapshotId: { type: Type.STRING },
              vehicleSimilarity: { type: Type.NUMBER },
              driverSimilarity: { type: Type.NUMBER },
              overallScore: { type: Type.NUMBER },
              isMatch: { type: Type.BOOLEAN },
              status: { type: Type.STRING, description: "VERIFIED, MISMATCH, or UNKNOWN" },
              reason: { type: Type.STRING }
            },
            required: ["id", "isMatch", "status", "entrySnapshotId", "exitSnapshotId", "vehicleSimilarity", "driverSimilarity", "overallScore"]
          }
        }
      }
    });

    return JSON.parse(response.text || "[]");
  } catch (error) {
    console.error("Gemini Analysis Error:", error);
    throw error;
  }
};
