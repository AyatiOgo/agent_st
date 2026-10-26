import logging
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import *
from .serializers import *
from agent_app.services.answer import answer_handbook_question
from django.shortcuts import get_object_or_404
from rest_framework import status


# Create your views here.


class TestView(APIView):
    def get(self, request):

        return Response({
            'message':'Hello This endpoint is working',
            'status': status.HTTP_200_OK
        })
        

logger = logging.getLogger(__name__)


class AskQuestionView(APIView):

    def post(self, request):
        # 1. Validate the student's question
        serializer = AskQuestionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        question = serializer.validated_data["question"]

        # 2. Select the latest completed handbook
        handbook = HandbookModel.objects.filter(
            status="completed"
        ).order_by("-updated_at", "-id").first()

        if handbook is None:
            return Response(
                {
                    "detail": (
                        "The admissions handbook is not available "
                        "right now."
                    )
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        # 3. Generate an answer using ChromaDB and OpenRouter
        try:
            result = answer_handbook_question(
                question=question,
                handbook_id=handbook.id,
            )
        except Exception:
            logger.exception("Failed to answer handbook question.")

            return Response(
                {
                    "detail": "Unable to answer the question right now."
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        # 4. Return the answer and page citations
        return Response(
            {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
            },
            status=status.HTTP_200_OK,
        )
