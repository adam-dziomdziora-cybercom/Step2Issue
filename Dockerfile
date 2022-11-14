#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.

FROM mcr.microsoft.com/dotnet/runtime:6.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:7.0.100 AS build
WORKDIR /src
COPY ["Step2Issue.csproj", "."]
RUN dotnet restore "./Step2Issue.csproj"
COPY . .
WORKDIR "/src/."
RUN dotnet build "Step2Issue.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "Step2Issue.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "Step2Issue.dll"]